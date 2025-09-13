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

def process_dataset_in_parallel(batch, tokenizer, max_length=2048):
    """
    并行处理函数：将构建prompt和tokenize合并，以充分利用多核CPU。
    'batch' 是一个批次的数据，格式为字典。
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
    # 注意：这里不直接tokenize，以保持与原逻辑一致，先生成文本
    texts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        ) for messages in messages_list
    ]
    
    # 3. 批量进行tokenization
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=False,
        max_length=max_length,
        return_tensors=None
    )
    
    # 4. 标签就是输入的copy
    tokenized['labels'] = tokenized['input_ids'].copy()
    
    return tokenized

def setup_model_and_tokenizer(model_path):
    """从本地路径设置模型和tokenizer"""
    print(f"从本地路径加载模型: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right"
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
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

def fine_tune_model(xlsx_path, model_path="/model/ModelScope/Qwen/Qwen3-8B", output_dir="./qwen_fine_tuned_model"):
    """主要的微调函数"""
    
    print("加载Qwen模型和tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(model_path)
    
    print("应用LoRA配置...")
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.train()
    
    # --- 数据处理流程重构 ---
    print("加载数据...")
    df = pd.read_excel(xlsx_path)
    if 'question' not in df.columns or 'answer' not in df.columns:
        raise ValueError("Excel文件必须包含'question'和'answer'两列")
    dataset = Dataset.from_pandas(df)
    print(f"加载了 {len(dataset)} 条原始数据")

    print("开始并行处理和Tokenize数据...")
    process_fn = partial(process_dataset_in_parallel, tokenizer=tokenizer, max_length=2048)

    tokenized_dataset = dataset.map(
        process_fn,
        batched=True,
        batch_size=1000,
        num_proc=4,  # 使用4个CPU核心并行处理
        remove_columns=dataset.column_names,
        desc="Processing and Tokenizing dataset"
    )
    # --- 数据处理流程结束 ---

    if len(tokenized_dataset) > 10:
        train_test_split = tokenized_dataset.train_test_split(test_size=0.1)
        train_dataset = train_test_split['train']
        eval_dataset = train_test_split['test']
    else:
        train_dataset = tokenized_dataset
        eval_dataset = None
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        warmup_steps=100,
        learning_rate=5e-5,
        weight_decay=0.01,
        bf16=torch.cuda.is_available(),
        logging_steps=10,
        logging_dir=f'{output_dir}/logs',
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=100 if eval_dataset else None,
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        report_to=None,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        remove_unused_columns=True,
        ddp_find_unused_parameters=False,
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
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