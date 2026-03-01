import click
import yaml
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv
from align_lab.inference.engine import run_inference, run_offline_inference
from align_lab.inference.judge import run_vllm_judge_inference
from align_lab.inference.raw_judge import run_raw_judge_inference
# from align_lab.eval.benchmarks import run_evaluation


load_dotenv()


def _load_sampling_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise click.BadParameter(f"Config file not found: {config_path}")

    data = yaml.safe_load(path.read_text())
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise click.BadParameter(f"Config file must be a mapping/dict: {config_path}")
    return data


class ProjectConfig(BaseModel):
    model_path: str = "meta-llama/Llama-3.1-70B-Instruct"  # Use Instruct model for chat inference
    device: str = "cuda"
    batch_size: int = 1


@click.group()
def cli():
    """AlignLab: Alleviate toxic self preference in LLMs with DPO Training"""
    pass


@cli.command()
@click.option("--config", type=click.Path(exists=True), help="Path to dpo_config.yaml")
@click.option("--lr", type=float, help="Override learning rate")
def train(config, lr):
    """Start DPO Training"""
    params = yaml.safe_load(Path(config).read_text())
    if lr:
        params["learning_rate"] = lr

    click.echo(f"Starting DPO Training on {params.get('model_path')}")
    # run_dpo(params)


@cli.command()
@click.option("--checkpoint", required=True, help="Model checkpoint to eval")
@click.option("--dataset", default="gsm8k", help="Eval dataset name")
def eval(checkpoint, dataset):
    """Run Benchmarks (GSM8K, MMLU, etc.)"""
    click.echo(f"Evaluating {checkpoint} on {dataset}...")
    # run_evaluation(checkpoint, dataset)


@cli.command()
@click.option("--model", required=True, help="Model ID (e.g., meta-llama/Llama-3.1-70B-Instruct for chat inference, or HF path)")
@click.option("--backend", type=click.Choice(["vllm", "openai", "anthropic", "deepseek"]), default="vllm", help="Inference backend engine")
@click.option("--api-key", envvar="LLM_API_KEY", help="API Key for OpenAI/Claude/DeepSeek")
@click.option("--base-url", help="Custom Base URL for API calls (useful for DeepSeek/LocalAI)")
@click.option("--hf-token", envvar="HF_TOKEN", help="HuggingFace Token (only for vLLM)")
@click.option(
    "--quantization",
    type=click.Choice(["bitsandbytes", "gptq", "awq", "fp8", "none"]),
    default="bitsandbytes",
    help="Quantization method for vLLM (default: bitsandbytes 8-bit)",
)
@click.option(
    "--data-path",
    default="data/01_processed_quality/quality_train.jsonl",
    help="Input JSONL file",
)
@click.option(
    "--output-path", default="outputs/quality_generations.jsonl", help="Output file"
)
@click.option(
    "--config-path",
    default="configs/inference.yaml",
    help="YAML config file for inference sampling hyperparameters",
)
def inference(model, backend, api_key, base_url, hf_token, quantization, data_path, output_path, config_path):
    """Run inference using vLLM (local) or API (GPT/Claude/DeepSeek)"""
    sampling_config = _load_sampling_config(config_path)
    
    if backend == "vllm":
        if not hf_token:
            click.echo("Warning: HF_TOKEN is usually required for gated models in vLLM.")
        quant = quantization if quantization != "none" else None
        run_offline_inference(
            model,
            data_path,
            output_path,
            hf_token,
            quantization=quant,
            sampling_config=sampling_config,
        )
    else:
        # Import here to avoid soft dependency issues if api modules are generic
        from align_lab.inference.engine import run_api_inference
        
        if not api_key:
             # DeepSeek/OpenAI/Anthropic need key unless using a local OpenAI-compatible server without auth
            if not base_url or "localhost" not in base_url:
                 click.echo(f"Warning: No --api-key provided for {backend}. Ensure it's set if required.")

        click.echo(f"Running API inference with {backend} model: {model}")
        run_api_inference(
            model_name=model,
            backend=backend,
            api_key=api_key,
            base_url=base_url,
            data_path=data_path,
            output_path=output_path,
            sampling_config=sampling_config,
        )


@cli.command()
@click.option("--model", required=True, help="Judge model ID (Instruct/chat model for vLLM)")
@click.option("--hf-token", envvar="HF_TOKEN", help="HuggingFace token for gated models")
@click.option(
    "--quantization",
    type=click.Choice(["bitsandbytes", "gptq", "awq", "fp8", "none"]),
    default="bitsandbytes",
    help="Quantization method for vLLM (default: bitsandbytes 8-bit)",
)
@click.option(
    "--data-path",
    default="data/04_clean_pairwise_output/llama31_70b__ds_v3.jsonl",
    help="Input pairwise-clean JSONL file",
)
@click.option(
    "--quality-path",
    default="data/01_processed_quality/quality_train.jsonl",
    help="QuALITY train JSONL used to recover article/question by question_unique_id",
)
@click.option(
    "--output-path",
    default="outputs/judge_results.jsonl",
    help="Output JSONL file (keeps all original fields + judge_output)",
)
@click.option(
    "--swap-answers",
    is_flag=True,
    help="Swap answer1/answer2 when building judge prompt",
)
@click.option(
    "--config-path",
    default="configs/judge.yaml",
    help="YAML config file for judge sampling hyperparameters",
)
def judge(model, hf_token, quantization, data_path, quality_path, output_path, swap_answers, config_path):
    """Run local vLLM judge on pairwise-clean outputs using article/question context."""
    sampling_config = _load_sampling_config(config_path)

    if not hf_token:
        click.echo("Warning: HF_TOKEN is usually required for gated models in vLLM.")

    quant = quantization if quantization != "none" else None

    click.echo(f"Running local judge with model: {model}")
    run_vllm_judge_inference(
        model_path=model,
        data_path=data_path,
        output_path=output_path,
        quality_path=quality_path,
        swap_answers=swap_answers,
        hf_token=hf_token,
        quantization=quant,
        sampling_config=sampling_config,
    )


@cli.command("raw_judge")
@click.option("--model", required=True, help="Judge model ID (Instruct/chat model)")
@click.option("--hf-token", envvar="HF_TOKEN", help="HuggingFace token for gated models")
@click.option(
    "--data-path",
    default="data/04_clean_pairwise_output/llama31_70b__ds_v3.jsonl",
    help="Input pairwise-clean JSONL file",
)
@click.option(
    "--quality-path",
    default="data/01_processed_quality/quality_train.jsonl",
    help="QuALITY train JSONL used to recover article/question by question_unique_id",
)
@click.option(
    "--output-path",
    default="outputs/raw_judge_results.jsonl",
    help="Output JSONL file (keeps all original fields + judge_output, prob_1, prob_2)",
)
@click.option(
    "--swap-answers",
    is_flag=True,
    help="Swap answer1/answer2 when building judge prompt",
)
@click.option(
    "--batch-size",
    default=1,
    type=int,
    help="Batch size for forward pass (default: 1)",
)
def raw_judge(model, hf_token, data_path, quality_path, output_path, swap_answers, batch_size):
    """Run judge via a single forward pass (transformers, bf16, no quantization).

    Instead of generating text, compares the probability of token "1" vs "2"
    at the last position of the model's output distribution.
    """
    if not hf_token:
        click.echo("Warning: HF_TOKEN is usually required for gated models.")

    click.echo(f"Running raw judge (forward-pass) with model: {model}")
    run_raw_judge_inference(
        model_path=model,
        data_path=data_path,
        output_path=output_path,
        quality_path=quality_path,
        swap_answers=swap_answers,
        hf_token=hf_token,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    cli()
