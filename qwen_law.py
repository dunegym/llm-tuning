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
import json

def load_and_prepare_data(xlsx_path, tokenizer):
    """加载xlsx文件并准备数据"""
    df = pd.read_excel(xlsx_path)
    
    # 确保列名正确
    if 'question' not in df.columns or 'answer' not in df.columns:
        raise ValueError("Excel文件必须包含'question'和'answer'两列")
    
    # 使用tokenizer自动生成对话格式
    texts = []
    for _, row in df.iterrows():
        # 构建对话消息格式
        messages = [
            {"role": "user", "content": str(row['question'])},
            {"role": "assistant", "content": str(row['answer'])}
        ]
        
        # 使用tokenizer的apply_chat_template自动生成prompt
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)
    
    return texts

def tokenize_function(examples, tokenizer, max_length=2048):
    """数据tokenization - 修复batch处理问题"""
    # 对文本进行tokenization，不返回tensor
    tokenized = tokenizer(
        examples['text'],
        truncation=True,
        padding=False,  # 先不padding，让data collator处理
        max_length=max_length,
        return_tensors=None  # 不返回tensor，返回list
    )
    
    # 对于因果语言模型，标签就是输入的copy
    tokenized['labels'] = tokenized['input_ids'].copy()
    
    return tokenized

def setup_model_and_tokenizer(model_path):
    """从本地路径设置模型和tokenizer"""
    print(f"从本地路径加载模型: {model_path}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right"
    )
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        attn_implementation="flash_attention_2"  # 修正：使用新的标准参数启用Flash Attention 2
    )
    
    # 确保有pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 准备模型进行训练 - 这很重要！
    model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer

def setup_lora_config():
    """设置LoRA配置 - 适配Qwen模型"""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,  # 降低rank以减少显存占用
        lora_alpha=32, # lora_alpha通常是r的两倍
        lora_dropout=0.1,
        bias="none",
        # Qwen模型的attention层名称
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    return lora_config

def fine_tune_model(xlsx_path, model_path="/model/ModelScope/Qwen/Qwen3-8B", output_dir="./qwen_fine_tuned_model"):
    """主要的微调函数"""
    
    # 1. 先设置模型和tokenizer
    print("加载Qwen模型和tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(model_path)
    
    # 2. 应用LoRA（在数据处理前应用LoRA）
    print("应用LoRA配置...")
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)
    
    # 打印可训练参数数量
    model.print_trainable_parameters()
    
    # 确保模型参数可训练
    model.train()
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    
    # 3. 加载数据（传入tokenizer用于生成prompt）
    print("加载数据...")
    texts = load_and_prepare_data(xlsx_path, tokenizer)
    print(f"加载了 {len(texts)} 条训练数据")
    
    # 打印第一个样本查看格式
    print(f"样本格式预览:\n{texts[0][:200]}...")
    
    # 4. 创建Dataset
    dataset = Dataset.from_dict({"text": texts})
    
    # 5. Tokenize数据
    print("处理数据...")
    def tokenize_function_wrapper(examples):
        return tokenize_function(examples, tokenizer, max_length=2048)
    
    tokenized_dataset = dataset.map(
        tokenize_function_wrapper, 
        batched=True,
        batch_size=1000,
        num_proc=4,  # 使用4个CPU核心并行处理数据
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )
    
    # 6. 分割训练和验证集
    if len(tokenized_dataset) > 10:
        train_test_split = tokenized_dataset.train_test_split(test_size=0.1)
        train_dataset = train_test_split['train']
        eval_dataset = train_test_split['test']
    else:
        train_dataset = tokenized_dataset
        eval_dataset = None
    
    # 7. 设置训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=1,  # 降低batch size适应显存
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,  # 增加梯度累积步数以补偿batch_size
        warmup_steps=100,
        learning_rate=5e-5,  # 降低学习率
        weight_decay=0.01,
        bf16=torch.cuda.is_available(),  # 使用bf16代替fp16
        logging_steps=10,
        logging_dir=f'{output_dir}/logs',
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=100 if eval_dataset else None,
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        report_to=None,
        dataloader_pin_memory=True, # 锁定内存以加速数据传输
        dataloader_num_workers=4, # 使用4个worker加载数据
        gradient_checkpointing=True,  # 启用梯度检查点以节省显存
        gradient_checkpointing_kwargs={'use_reentrant': False}, # 推荐与PEFT一起使用
        remove_unused_columns=True, # 推荐设置为True
        ddp_find_unused_parameters=False,
    )
    
    # 8. 设置数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )
    
    # 9. 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,  # 使用新的参数名
    )
    
    # 10. 开始训练
    print("开始微调...")
    trainer.train()
    
    # 11. 保存模型
    print(f"保存模型到 {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print("微调完成!")
    
    return model, tokenizer

def test_model(model_path, question):
    """测试微调后的模型"""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 使用tokenizer自动生成prompt格式
    messages = [{"role": "user", "content": question}]
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
    
    # 生成回答
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=512,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 解码输出
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # 提取新生成的部分
    answer = generated_text[len(input_text):].strip()
    
    return answer

if __name__ == "__main__":
    # 使用示例
    xlsx_file_path = "/cloud/cloud-ssd1/llm/law_qa.xlsx"  # 替换为您的xlsx文件路径
    model_path = "/model/ModelScope/Qwen/Qwen3-4B"  # 本地模型路径

    try:
        # 进行微调
        model, tokenizer = fine_tune_model(
            xlsx_path=xlsx_file_path,
            model_path=model_path,
            output_dir="./qwen_fine_tuned_model"
        )
        
        # 测试模型
        test_question = "什么是合同法？"
        answer = test_model("./qwen_fine_tuned_model", test_question)
        print(f"问题: {test_question}")
        print(f"回答: {answer}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()