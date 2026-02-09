import click
import yaml
from pathlib import Path
from pydantic import BaseModel
# from align_lab.training.dpo_trainer import run_dpo
from align_lab.inference.engine import run_inference, run_offline_inference
# from align_lab.eval.benchmarks import run_evaluation


class ProjectConfig(BaseModel):
    model_path: str = "path/to/llama-3.1-70b"
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
@click.option("--model", required=True, help="Model ID (e.g., gpt-3.5-turbo, or HF path)")
@click.option("--backend", type=click.Choice(["vllm", "openai", "anthropic", "deepseek"]), default="vllm", help="Inference backend engine")
@click.option("--api-key", envvar="LLM_API_KEY", help="API Key for OpenAI/Claude/DeepSeek")
@click.option("--base-url", help="Custom Base URL for API calls (useful for DeepSeek/LocalAI)")
@click.option("--hf-token", envvar="HF_TOKEN", help="HuggingFace Token (only for vLLM)")
@click.option(
    "--data-path",
    default="data/01_processed_quality/quality_train.jsonl",
    help="Input JSONL file",
)
@click.option(
    "--output-path", default="outputs/quality_generations.jsonl", help="Output file"
)
def inference(model, backend, api_key, base_url, hf_token, data_path, output_path):
    """Run inference using vLLM (local) or API (GPT/Claude/DeepSeek)"""
    
    if backend == "vllm":
        if not hf_token:
            click.echo("Warning: HF_TOKEN is usually required for gated models in vLLM.")
        run_offline_inference(model, data_path, output_path, hf_token)
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
            output_path=output_path
        )


if __name__ == "__main__":
    cli()
