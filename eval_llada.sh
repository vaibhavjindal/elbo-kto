set -x
set -e


pip install transformers==4.49.0 lm_eval==0.4.9 accelerate==0.34.2
pip install antlr4-python3-runtime math_verify sympy hf_xet


# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

# Change this to your model path
LLADA_INSTRUCT=GSAI-ML/LLaDA-8B-Instruct


# mmlu_generative
accelerate launch eval_llada.py \
    --apply_chat_template \
    --tasks mmlu_generative \
    --model llada_dist \
    --num_fewshot 5 \
    --model_args model_path=$LLADA_INSTRUCT,gen_length=3,steps=3,block_length=3


# gsm8k
accelerate launch eval_llada.py \
    --apply_chat_template \
    --tasks gsm8k \
    --model llada_dist \
    --num_fewshot 5 \
    --model_args model_path=$LLADA_INSTRUCT,gen_length=256,steps=256,block_length=8


#humaneval
accelerate launch eval_llada.py \
    --tasks humaneval_instruct_sanitized \
    --include_path tasks/humaneval \
    --apply_chat_template \
    --model llada_dist \
    --model_args model_path=$LLADA_INSTRUCT,gen_length=512,steps=512,block_length=32 \
    --confirm_run_unsafe_code