from pprint import pprint
from ab.gpt.util.prompt.NNGenPrompt import NNGenPrompt
from ab.gpt.util.prompt.NNRLPrompt import NNRLPrompt
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.grpo_trainer import GRPOTrainer
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.sft_config import SFTConfig
from datasets import Dataset
from ab.gpt.util.Const import conf_train_dir, conf_test_dir
from ab.nn.util.Exception import AccuracyException
import ab.nn.api as api
from ab.gpt.util.Util import extract_code
import ast
import json


def reward_fn(code: str) -> float:
    # if "[MASKED LAYER]" in code or "MASKED" in code:
    #     return -1.0
    try:
        res = api.check_nn(code, task='img-classification', 
                           dataset='cifar-10', metric='acc', prm={'lr': 0.01, 'batch': 16, 'dropout': 0.3, 'momentum': 0.9,
                   'transform': 'norm_256_flip', 'epoch': 1}, save_to_db=False)
        return res[3]
    # except AccuracyException:
    #     return res[0]
    except Exception as e:
        print("Eval failed", e)
        return 0.0
    
    # reward function
def compute_reward(prompts, completions, **kwargs):
    """
    prompts: List[str]
    completions: List[str]
    Returns: List[float] as rewards
    """
    rewards = []
    for prompt, completion in zip(prompts, completions):
        print("=" * 50)
        print("Raw completion:")
        print(completion)
        print("=" * 50)
        
        # 使用extract_code方法提取代码
        extracted_code = extract_code(completion)
        print("Extracted code:")
        print(repr(extracted_code))  # 使用repr显示None或实际内容
        print("=" * 50)
        
        # 如果extract_code返回None，尝试其他方法提取
        if extracted_code is None or extracted_code.strip() == "":
            # 查找代码块模式
            import re
            code_patterns = [
                r'```python\s*\n(.*?)\n```',
                r'```\s*\n(.*?)\n```',
                r'self\.[^\n]+nn\.[^\n]+',  # 匹配self.xxx = nn.xxx形式的代码
            ]
            
            for pattern in code_patterns:
                matches = re.findall(pattern, completion, re.DOTALL)
                if matches:
                    extracted_code = matches[0] if isinstance(matches[0], str) else '\n'.join(matches)
                    print(f"Found code using pattern: {pattern}")
                    print(f"Extracted: {repr(extracted_code)}")
                    break
        
        # 如果仍然没有代码，给个默认值避免错误
        if extracted_code is None or extracted_code.strip() == "":
            print("No valid code found, using empty string")
            extracted_code = ""
        
        # 评估代码质量
        try:
            score = reward_fn(extracted_code) if extracted_code.strip() else 0.0
            print(f"Reward score: {score}")
        except Exception as e:
            print("Eval failed", e)
            score = 0.0
        
        rewards.append(score)
        
        # 只处理第一个样本，然后退出
        print("Processing first sample only, exiting...")
        import sys
        sys.exit(0)
        
    return rewards

def clean_code_text(text):
    lines = text.strip().split('\n')
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return '\n'.join(lines).strip()

# formalize the data
def format_for_rl(example):
    # instruction = example["instruction"] if example["instruction"] else ""
    # context = example["context"] if example["context"] else ""
    # prompt = instruction.strip()
    prompt = example["prompt"] if example["prompt"] else ""
    response = example["response"] if example["response"] else ""
    # response = clean_code_text(response['nn_code'])
    return {
        "prompt": prompt,
        "response": response.strip(),
        "completion": response.strip(),
    }

# trainer.save_model("./grpo_outputs/final_checkpoint")
if __name__ == "__main__":
        
    # configing the model
    # train_config_path = conf_train_dir / 'Mix copy.json'
    train_config_path = conf_test_dir / 'NN_gen.json'
    base_model = "deepseek-ai/deepseek-coder-1.3b-instruct"  # 使用更小的模型
    # base_model = "deepseek-ai/deepseek-coder-6.7b-instruct"
    # base_model = "ABrain/NNGPT-DeepSeek-Coder-1.3B-Instruct""ABrain/HPGPT-DeepSeek-R1-Distill-Qwen-7B-R"

    # tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    # tokenizer and model
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_eos_token = True
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    
    # 添加停止标记来控制生成
    stop_tokens = ["```", "def ", "class ", "import ", "from ", "# ", "```python"]
    stop_token_ids = []
    for token in stop_tokens:
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        if token_ids:
            stop_token_ids.extend(token_ids)

    # M1 Mac优化：禁用BitsAndBytes量化，使用原生MPS
    if torch.backends.mps.is_available():
        print("使用MPS设备，禁用BitsAndBytes量化")
        model = AutoModelForCausalLM.from_pretrained(
            base_model, 
            trust_remote_code=True, 
            torch_dtype=torch.float32,  # 使用32位提高数值稳定性
            device_map="mps"
        )
        # 为MPS设备添加数值稳定性设置
        model.config.use_cache = False  # 禁用KV缓存避免内存问题
        if hasattr(model.config, 'torch_dtype'):
            model.config.torch_dtype = torch.float32
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16, 
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model, 
            trust_remote_code=True, 
            quantization_config=bnb_config,
            device_map="auto"
        )

    # LoRA - 更保守的配置，基于成功的简化SFT测试
    peft_config = LoraConfig(
        r=4,  # 降低rank避免过拟合
        lora_alpha=8,  # 对应r=4的alpha值
        target_modules=["q_proj", "v_proj"],  # 只训练查询和值投影，更保守
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # data processing
    # data_processor = NNPrompt(100000, tokenizer)
    
    # data_processor = NNRLPrompt(100000, tokenizer, train_config_path)

    data_processor = NNGenPrompt(100000, tokenizer, train_config_path)
    raw_df = data_processor.get_raw_dataset(False, n_training_prompts=100000)
    clean_df = raw_df[["instruction", "response", "text"]]
    dataset = Dataset.from_pandas(clean_df)
    dataset.save_to_disk("dataset.json")

    # dataset = Dataset.load_from_disk("dataset.json")
    # 大幅减少训练数据量，基于成功的简化SFT测试
    dataset = dataset.select(range(20)).shuffle()  # 只使用20个高质量样本
#     PROMPT_TEMPLATE = [
#     "Below is a partial implementation of a PyTorch neural network.",
#     "Some layers in the `__init__` method are replaced with the placeholder `[MASKED LAYER]`.",
#     # "Here is the beginning of the implementation:",
#     "",
#     "{masked_prompt}",
#     "",
#     # "Your task is to complete the layer definitions in the `__init__` method. Replace the # [MASKED LAYER], DO NOT change anything else."
#     "Your task is to generate the layer definitions, DO NOT change anything else."
#     # "Your task is to ONLY replace the [MASKED LAYER] in the `__init__` method."
#     # "Strictly DO NOT change or rewrite any other part of the code.",
#     # "The class name MUST be `Net`",
#     # "The code must include all necessary methods like `__init__`, `forward`, `train_setup`, and `learn`.",
#     "Now output the layer definitions below:"
# ]
    PROMPT_TEMPLATE = [
    "Generate a PyTorch neural network layer based on this task:",
    "",
    "{masked_prompt}",
    "",
    "Requirements:",
    "1. Must be valid Python/PyTorch code",
    "2. Include necessary imports (torch, torch.nn)",
    "3. Define a class inheriting from nn.Module",
    "4. Implement __init__ and forward methods",
    "5. Output as a complete code block between ```python and ```",
    "",
    "Example format:",
    "```python",
    "import torch",
    "import torch.nn as nn",
    "",
    "class Net(nn.Module):",
    "    def __init__(self):",
    "        super(Net, self).__init__()",
    "        # layer definitions here",
    "        ",
    "    def forward(self, x):",
    "        # forward pass here", 
    "        return x",
    "```"
]
    def mask_layer_definitions(code: str) -> tuple[str, str]:
        try:
            tree = ast.parse(code)
            lines = code.splitlines()
            masked_lines = list(lines)
            extracted_layers = []

            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if (
                            isinstance(target, ast.Attribute)
                            and isinstance(target.value, ast.Name)
                            and target.value.id == "self"
                        ):
                            if (
                                isinstance(node.value, ast.Call)
                                and isinstance(node.value.func, ast.Attribute)
                                and node.value.func.value.id in {"nn", "torch.nn"}
                            ):
                                start_line = node.lineno - 1
                                extracted_layers.append(lines[start_line])
                                masked_lines[start_line] = "# [MASKED LAYER]"

            return "\n".join(masked_lines), "\n".join(extracted_layers)
        except Exception:
            return code, ""

    def build_masked_dataset(example):
        code = example["response"]
        parsed = json.loads(code)

        code = parsed["nn_code"]
        masked_model, masked_code = mask_layer_definitions(code)
        return {
            "prompt" : "\n".join([
                line if "{masked_prompt}" not in line else line.format(masked_prompt=masked_model)
                for line in PROMPT_TEMPLATE
            ]),
            "response": masked_code.strip()
            # "response": code.strip()
        }

    masked_dataset = dataset.map(build_masked_dataset, remove_columns=dataset.column_names)
    dataset = masked_dataset.filter(lambda x: x["response"] != "")
    # pprint(dataset[0])


    dataset = dataset.map(format_for_rl, remove_columns=dataset.column_names)
    # dataset = dataset.filter(lambda x: len(x["response"]) < 10000)

    print("Dataset size:", len(dataset))
    # pprint(dataset[0])
    # pprint(extract_code(dataset[0]))
    # SFT
    def chat_fn(x):
        messages = [{"role": "user", "content": x["prompt"]},
                    {"role": "assistant", "content": x["response"]}]
        prompt_str = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=False                
        )
        return {"text": prompt_str}
    # 使用更少的数据进行SFT训练，基于实际可用数据量
    available_size = len(dataset)
    sample_size = min(10, available_size)  # 取可用数据和10的最小值
    print(f"可用数据: {available_size}个样本，使用: {sample_size}个样本")
    
    tokenized_dataset = dataset.select(range(sample_size))  # 根据实际情况选择样本数
    
    # 简化数据格式，基于成功的测试
    def format_for_sft(example):
        return {
            "prompt": example["prompt"],
            "completion": example["response"]
        }
    
    tokenized_dataset = tokenized_dataset.map(format_for_sft, remove_columns=tokenized_dataset.column_names)
    # 非常保守的SFT配置，基于成功的简化测试
    sft_config = TrainingArguments(
        output_dir="./sft_outputs",
        num_train_epochs=1,  # 只训练1轮避免过拟合
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,  # 减少梯度累积
        warmup_steps=2,  # 减少warmup步数
        logging_steps=1,
        save_steps=999,  # 不保存中间checkpoint
        save_strategy="no",  # 不保存模型
        load_best_model_at_end=False,
        gradient_checkpointing=True,
        fp16=False,  # M1 Mac 使用float32
        learning_rate=1e-5,  # 很小的学习率
        weight_decay=0.01,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        report_to=[]  # 禁用wandb
    )

    sft_trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=sft_config,
        peft_config=peft_config
    )    
    
    # 启用SFT训练
    print("开始保守的SFT训练...")
    print("修正要点:")
    print("1. 使用很少的高质量数据 (10个样本)")
    print("2. 保守的LoRA设置 (r=4, 只训练q_proj和v_proj)")
    print("3. 很小的学习率 (1e-5)")
    print("4. 短时间训练 (1个epoch)")
    print("这些修正基于成功的简化SFT测试结果")
    print("="*50)
    
    sft_trainer.train()
    
    # 保存SFT模型
    # sft_trainer.save_model("./sft_outputs/final_checkpoint")
    # print("SFT训练完成，模型已保存到 ./sft_outputs/final_checkpoint")
    
    # 测试SFT训练后的效果
    print("\n" + "="*50)
    print("测试SFT训练后的生成效果:")
    print("="*50)
    
    # 简单的测试prompt
    test_prompt = "请用PyTorch创建一个ReLU激活层"
    
    print(f"测试prompt: {test_prompt}")
    print("-" * 30)
    
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,  # 减少生成长度
            do_sample=True,
            temperature=0.3,  # 降低温度获得更稳定输出
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    print("生成结果:")
    print(generated_text)
    print("\n" + "="*50)
    print("提取的代码:")
    extracted = extract_code(generated_text)
    if extracted:
        print(extracted)
    else:
        print("未能提取到有效代码")
    
    print("\n训练完成！基于成功的简化SFT方法修正。")
    print("如果效果良好，可以考虑进行RL训练。")
    
    # 退出，不继续RL训练（除非用户明确需要）
    import sys
    sys.exit(0)



    # dataset = dataset.select(range(500))
    def to_rl_prompt(ex):
        messages = [{"role": "user", "content": ex["prompt"]}]
        prompt_str = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True,   # 明确回答起点
            # return_tensors="pt",
            tokenize=False                # 返回字符串
        )
        return {"prompt": prompt_str}

    # # 2) 只保留 prompt 列，去掉其他所有列
    rl_dataset = dataset.map(to_rl_prompt, remove_columns=dataset.column_names)

    # sample = dataset[0]
    # messages=[
    #     { 'role': 'user', 'content': sample["prompt"]},
    # ]
    # inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, )
    # pprint(inputs)
    # outputs = model.generate(inputs, max_new_tokens=8192, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
    # print(tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True))
    # print(reward_fn(extract_code(tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True))))
    # pprint(rl_dataset[0])
    # GRPOTrainer
    grpo_config = GRPOConfig(
        learning_rate=5e-5,
        max_completion_length=2048,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        lr_scheduler_type="cosine",
        num_train_epochs=3,
        remove_unused_columns=False,
        logging_steps=10,
        output_dir="./grpo_outputs",
        eval_strategy="no",
        bf16=True,
        # gradient_checkpointing=True,
        num_generations=2,
   
        generation_kwargs={
            "max_new_tokens": 512,      # 增加长度以允许完整代码块
            "do_sample": True,
            "top_p": 0.9,               # 恢复合理的随机性
            "top_k": 50,                # 增加选择范围
            "temperature": 0.7,         # 适中的温度
            "repetition_penalty": 1.1,  # 适中的重复惩罚
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            # MPS设备上的数值稳定性改进
            "renormalize_logits": True,
        },
        # use_peft=True
    )

    trainer = GRPOTrainer(
        # config=grpo_config,
        model=model,
        # tokenizer=tokenizer,
        train_dataset=rl_dataset,
        reward_funcs=compute_reward,
        args=grpo_config,
        processing_class=None
    )

    trainer.train()