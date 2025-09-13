import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from functools import partial
import os
import multiprocessing as mp

def process_dataset_in_parallel(batch, tokenizer, max_length=2048):
    """
    并行处理函数：将构建prompt和tokenize合并，以充分利用多核CPU。
    """
    # 1. 批量构建对话消息格式
    messages_list = []
    for question, answer in zip(batch['question'], batch['answer']):
        messages = [
            {"role": "user", "content": str(question)},
            {"role": "assistant", "content": str(answer)}
        ]
        messages_list.append(messages)

    # 2. 批量应用聊天模板
    texts = []
    for messages in messages_list:
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)
        except Exception as e:
            print(f"Warning: Failed to process message: {e}")
            continue
    
    if not texts:
        return {"input_ids": [], "attention_mask": [], "labels": []}
    
    # 3. 修复：进行tokenization但确保返回正确格式
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=False,  # 关闭padding
        max_length=max_length,
        return_tensors=None,
        verbose=False
    )
    
    # 4. 修复：确保labels是与input_ids相同的格式
    tokenized['labels'] = tokenized['input_ids'].copy()  # 直接复制
    
    return tokenized

def setup_model_and_tokenizer(model_path):
    """从本地路径设置模型和tokenizer - 优化版本"""
    print(f"从本地路径加载模型: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right",
        use_fast=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer

def setup_lora_config():
    """设置LoRA配置"""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

class CustomDataCollatorForCausalLM:
    """自定义数据整理器，确保正确处理padding和多核兼容"""
    
    def __init__(self, tokenizer, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        batch = {}
        
        # 获取最大长度
        max_length = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            max_length = ((max_length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        
        for f in features:
            input_ids = f["input_ids"]
            attention_mask = f["attention_mask"] 
            labels = f["labels"]
            
            # 手动padding
            padding_length = max_length - len(input_ids)
            
            # Pad input_ids
            padded_input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            batch_input_ids.append(padded_input_ids)
            
            # Pad attention_mask
            padded_attention_mask = attention_mask + [0] * padding_length
            batch_attention_mask.append(padded_attention_mask)
            
            # Pad labels (用-100填充padding位置)
            padded_labels = labels + [-100] * padding_length
            batch_labels.append(padded_labels)
        
        batch["input_ids"] = torch.tensor(batch_input_ids, dtype=torch.long)
        batch["attention_mask"] = torch.tensor(batch_attention_mask, dtype=torch.long)
        batch["labels"] = torch.tensor(batch_labels, dtype=torch.long)
        
        return batch

def fine_tune_model(xlsx_path, model_path="/model/ModelScope/Qwen/Qwen3-8B", output_dir="./qwen_fine_tuned_model"):
    """主要的微调函数 - 稳定优化版本"""
    
    # 关键优化：设置多线程环境变量
    cpu_count = mp.cpu_count()
    
    # 为不同组件设置合适的线程数
    os.environ["OMP_NUM_THREADS"] = str(cpu_count)
    os.environ["MKL_NUM_THREADS"] = str(cpu_count)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_count)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_count)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_count)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    
    # PyTorch 线程设置
    torch.set_num_threads(cpu_count)
    torch.set_num_interop_threads(cpu_count)
    
    print(f"设置CPU线程数: {cpu_count}")
    print("加载Qwen模型和tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(model_path)
    
    print("应用LoRA配置...")
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.train()
    
    # 数据处理流程
    print("加载数据...")
    df = pd.read_excel(xlsx_path)
    if 'question' not in df.columns or 'answer' not in df.columns:
        raise ValueError("Excel文件必须包含'question'和'answer'两列")
    
    df = df.dropna(subset=['question', 'answer'])
    dataset = Dataset.from_pandas(df)
    print(f"加载了 {len(dataset)} 条原始数据")

    print("开始并行处理和Tokenize数据...")
    process_fn = partial(process_dataset_in_parallel, tokenizer=tokenizer, max_length=2048)

    batch_size = 100
    num_proc = max(1, min(4, cpu_count - 1))
    
    print(f"使用 {num_proc} 个CPU核心，批处理大小: {batch_size}")
    
    tokenized_dataset = dataset.map(
        process_fn,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc="Processing and Tokenizing dataset",
        writer_batch_size=1000,
        keep_in_memory=False,
    )

    print(f"Tokenized数据集大小: {len(tokenized_dataset)}")
    
    # 检查数据集格式
    print("检查tokenized数据集样本...")
    sample = tokenized_dataset[0]
    print(f"样本字段: {list(sample.keys())}")
    print(f"input_ids长度: {len(sample['input_ids'])}")
    print(f"labels长度: {len(sample['labels'])}")
    print(f"input_ids类型: {type(sample['input_ids'])}")
    print(f"labels类型: {type(sample['labels'])}")
    
    if len(tokenized_dataset) > 10:
        train_test_split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = train_test_split['train']
        eval_dataset = train_test_split['test']
    else:
        train_dataset = tokenized_dataset
        eval_dataset = None
    
    # 先尝试单进程数据加载器，确保稳定性
    print("使用稳定的训练配置...")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,  
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,  
        warmup_steps=100,
        learning_rate=5e-5,
        weight_decay=0.01,
        bf16=torch.cuda.is_available(),
        fp16=False,
        logging_steps=10,
        logging_dir=f'{output_dir}/logs',
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=100 if eval_dataset else None,
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        report_to=None,
        
        # 暂时使用单进程数据加载器确保稳定
        dataloader_pin_memory=False,
        dataloader_num_workers=0,  # 先设为0确保稳定
        
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        remove_unused_columns=True,
        ddp_find_unused_parameters=False,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        group_by_length=False,
        auto_find_batch_size=False,
    )
    
    # 使用自定义数据整理器
    data_collator = CustomDataCollatorForCausalLM(
        tokenizer=tokenizer,
        pad_to_multiple_of=8
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    print("开始微调...")
    trainer.train()
    
    print(f"保存模型到 {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print("微调完成!")
    return model, tokenizer

# test_model 函数保持不变...
def test_model(model_path, question):
    """测试微调后的模型"""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    messages = [{"role": "user", "content": question}]
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(model.device)
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=512,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # 简单地从用户输入之后开始截取
    answer_start_index = response_text.rfind(question) + len(question)
    answer = response_text[answer_start_index:].strip()
    
    return answer

if __name__ == "__main__":
    xlsx_file_path = "/cloud/cloud-ssd1/llm/law_qa.xlsx"
    model_path = "/model/ModelScope/Qwen/Qwen3-4B"

    try:
        model, tokenizer = fine_tune_model(
            xlsx_path=xlsx_file_path,
            model_path=model_path,
            output_dir="./qwen_fine_tuned_model"
        )
        
        test_question = "什么是合同法？"
        answer = test_model("./qwen_fine_tuned_model", test_question)
        print(f"\n--- 测试模型 ---\n问题: {test_question}\n回答: {answer}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()