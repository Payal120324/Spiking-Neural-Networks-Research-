"""
Spike-Legal-NLP Research Framework
A Semantic- and Energy-Aware Study of Spike Encoding for Legal Text
Classification Under Domain Shift

Usage:
  python main.py run                          # Full pipeline (default config)
  python main.py run --dataset case_hold      # Run on specific dataset
  python main.py run --encoder legal_bert     # Use specific transformer
  python main.py run --encodings poisson_rate latency  # Only these encodings
  python main.py run --skip snn              # Skip SNN training stage
  python main.py run --target-dataset ecthr_a # Also run domain-shift eval (RQ3/H4)

  python main.py dataset list                 # List available datasets
  python main.py dataset info case_hold       # Show dataset statistics
  python main.py dataset download case_hold   # Pre-download a dataset
  python main.py dataset delete case_hold     # Delete cached dataset

  python main.py encode demo                  # Demo spike encoding on sample text
  python main.py encode compare               # Compare all encodings visually

  python main.py report --results storage/results/results_*.json  # Regenerate report
"""

import logging
import sys
import os
from pathlib import Path

# Disable HuggingFace auto-conversion to avoid network issues
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '0'  # Still allow downloads, but disable background threads
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'  # Disable transfer for better compatibility
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'  # Disable symlinks warning on Windows
os.environ['HF_HUB_ENABLE_XET'] = '0'  # Disable XET for better compatibility

import click
import yaml

# ─────────────────────────────────────────────────────────────────────
# Project root is the directory containing this file
# ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# Storage directories (created on import)
for d in [
    "storage/datasets/raw",
    "storage/datasets/processed",
    "storage/datasets/cache",
    "storage/embeddings",
    "storage/checkpoints",
    "storage/results",
    "storage/results/figures",
    "storage/results/reports",
]:
    (ROOT / d).mkdir(parents=True, exist_ok=True)


def setup_logging(level: str = "INFO", quiet: bool = False):
    # Configure file handler with UTF-8 encoding to handle Unicode characters
    file_handler = logging.FileHandler("storage/results/experiment.log", encoding='utf-8')
    handlers: list[logging.Handler] = [file_handler]

    if not quiet:
        # Mirror progress to the console; reconfigure stdout to UTF-8 so the
        # box-drawing characters used in log messages work on Windows consoles.
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
        )
        handlers.append(console_handler)

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )
    # Quieten noisy libraries
    for noisy in [
        "transformers", "datasets", "huggingface_hub", "urllib3", "filelock",
        "httpx", "httpcore", "fsspec", "matplotlib",
    ]:
        logging.getLogger(noisy).setLevel(logging.WARNING)


def load_config(config_path: str = "config.yaml") -> dict:
    path = ROOT / config_path
    if not path.exists():
        path = Path(config_path)
    with open(path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

@click.group()
@click.option("--config", default="config.yaml", help="Path to config.yaml")
@click.option("--log-level", default="INFO", help="Logging level")
@click.option("--quiet", is_flag=True, help="Log to file only (no console output)")
@click.pass_context
def cli(ctx, config, log_level, quiet):
    """Spike-Legal-NLP: Research Framework for Spike Encoding in Legal NLP."""
    setup_logging(log_level, quiet=quiet)
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config)


# ─────────────────────────────────────────────────────────────────────
# run command
# ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--dataset", default=None, help="Dataset key (e.g. case_hold, ecthr_a)")
@click.option("--encoder", default=None, help="Transformer model key (e.g. legal_bert, bert)")
@click.option("--encodings", multiple=True, help="Spike encoding methods to use")
@click.option("--skip", multiple=True, help="Pipeline stages to skip")
@click.option("--quick", is_flag=True, help="Quick mode: reduced samples, fewer encodings")
@click.option("--seeds", default=1, type=int, help="Number of random seeds for multi-seed averaging (default: 1)")
@click.option("--max-train-samples", default=None, type=int, help="Cap training samples per dataset")
@click.option("--max-val-samples", default=None, type=int, help="Cap validation samples per dataset")
@click.option("--max-test-samples", default=None, type=int, help="Cap test samples per dataset")
@click.option(
    "--target-dataset", default=None,
    help="Second dataset key to evaluate zero-shot domain-shift transfer "
         "against (RQ3 / H4), e.g. ecthr_a. Omit to skip domain-shift evaluation.",
)
@click.pass_context
def run(ctx, dataset, encoder, encodings, skip, quick, seeds, max_train_samples,
        max_val_samples, max_test_samples, target_dataset):
    """Run the full experiment pipeline."""
    config = ctx.obj["config"]

    print("=" * 70)
    print("Spike-Legal-NLP Research Framework")
    print("A Semantic- and Energy-Aware Study of Spike Encoding for Legal Text Classification")
    print("=" * 70)

    if quick:
        config["datasets"]["max_train_samples"] = 100
        config["datasets"]["max_val_samples"] = 50
        config["datasets"]["max_test_samples"] = 100
        config["snn"]["training"]["num_epochs"] = 3
        config["models"]["training"]["num_epochs"] = 1
        print("Quick mode: reduced sample sizes")

    for opt_value, cfg_key in [
        (max_train_samples, "max_train_samples"),
        (max_val_samples, "max_val_samples"),
        (max_test_samples, "max_test_samples"),
    ]:
        if opt_value is not None:
            config["datasets"][cfg_key] = opt_value

    dataset = dataset or list(
        k for k, v in config.get("datasets", {}).get("available", {}).items()
        if v.get("active", False)
    )[0]
    encoder = encoder or next(
        (k for k, v in config.get("models", {}).get("transformers", {}).items()
         if v.get("active", True)),
        "legal_bert",
    )
    encodings = list(encodings) or None
    skip = list(skip)

    print(f"Dataset:     {dataset}")
    print(f"Transformer: {encoder}")
    print(f"Encodings:   {encodings or 'all enabled'}")
    print(f"Seeds:       {seeds}")
    ds_cfg = config.get("datasets", {})
    print(
        "Sample caps: train={}, val={}, test={}".format(
            ds_cfg.get("max_train_samples") or "all",
            ds_cfg.get("max_val_samples") or "all",
            ds_cfg.get("max_test_samples") or "all",
        )
    )
    print(f"Target (domain shift): {target_dataset or 'none (H4 skipped)'}")
    print()

    from src.experiments import ExperimentPipeline

    pipeline = ExperimentPipeline(config)
    results = pipeline.run(
        dataset_key=dataset,
        transformer_key=encoder,
        encodings=encodings,
        skip_stages=skip,
        target_dataset_key=target_dataset,
        num_seeds=seeds,
    )

    print("\n[OK] Pipeline complete!")
    if "report" in results:
        print(f"Report: {results['report']}")
    if "energy" in results:
        print("\nEnergy Summary:")
        for enc, er in results["energy"].items():
            if "mean" in er.get("comparison", {}):
                # Multi-seed results with mean+/-std
                comp = er["comparison"]
                print(
                    f"  {enc}: {comp['energy_ratio']['mean']:.1f}+/-{comp['energy_ratio']['std']:.1f}x efficiency, "
                    f"{comp['energy_savings_pct']['mean']:.1f}+/-{comp['energy_savings_pct']['std']:.1f}% savings"
                )
            else:
                # Single-seed results
                print(
                    f"  {enc}: {er['comparison']['energy_ratio']:.1f}x efficiency, "
                    f"{er['comparison']['energy_savings_pct']:.1f}% savings"
                )
    if results.get("domain_shift"):
        print("\nDomain Shift Summary:")
        shift = results["domain_shift"]
        for model_key in shift.get("_ranking_by_robustness", []):
            s = shift[model_key]
            print(
                f"  {model_key}: avg drop={s['avg_accuracy_drop']:.4f}, "
                f"avg target acc={s['avg_target_accuracy']:.4f}"
            )


# ─────────────────────────────────────────────────────────────────────
# dataset commands
# ─────────────────────────────────────────────────────────────────────

@cli.group()
def dataset():
    """Dataset management commands."""
    pass


@dataset.command("list")
@click.pass_context
def dataset_list(ctx):
    """List all available datasets."""
    from rich.console import Console
    from rich.table import Table
    from src.datasets import DatasetManager

    config = ctx.obj["config"]
    dm = DatasetManager(config)
    available = dm.list_available()

    table = Table(title="Available Legal NLP Datasets", show_header=True)
    table.add_column("Key", style="bold cyan")
    table.add_column("Task")
    table.add_column("Description")
    table.add_column("Cached?", justify="center")
    table.add_column("Cache Time")

    for ds in available:
        cached_str = "[green]Y[/green]" if ds["cached"] else "[red]N[/red]"
        table.add_row(
            ds["key"],
            ds.get("task", "—"),
            ds.get("description", "—"),
            cached_str,
            str(ds.get("cache_time", "—"))[:19] if ds.get("cache_time") else "—",
        )

    Console().print(table)


@dataset.command("info")
@click.argument("dataset_key")
@click.pass_context
def dataset_info(ctx, dataset_key):
    """Show statistics for a dataset (downloads if not cached)."""
    from src.datasets import DatasetManager, DatasetStatistics
    from src.datasets.manager import DATASET_REGISTRY

    config = ctx.obj["config"]
    dm = DatasetManager(config)
    data = dm.load(dataset_key)
    info = DATASET_REGISTRY.get(dataset_key, {})
    stats = DatasetStatistics(data, info)
    stats.compute_all()
    stats.print_summary()


@dataset.command("download")
@click.argument("dataset_key")
@click.option("--force", is_flag=True, help="Force re-download even if cached")
@click.pass_context
def dataset_download(ctx, dataset_key, force):
    """Pre-download and cache a dataset."""
    from rich.console import Console
    from src.datasets import DatasetManager

    config = ctx.obj["config"]
    dm = DatasetManager(config)
    Console().print(f"Downloading [cyan]{dataset_key}[/cyan]…")
    dm.load(dataset_key, force_download=force)
    Console().print(f"[green][OK] {dataset_key} cached successfully[/green]")


@dataset.command("delete")
@click.argument("dataset_key")
@click.pass_context
def dataset_delete(ctx, dataset_key):
    """Delete cached version of a dataset."""
    from rich.console import Console
    from src.datasets import DatasetManager

    config = ctx.obj["config"]
    dm = DatasetManager(config)
    dm.delete_cache(dataset_key)
    Console().print(f"[yellow]Deleted cache for {dataset_key}[/yellow]")


@dataset.command("custom")
@click.argument("file_path")
@click.option("--text-col", default=None, help="Column name for text")
@click.option("--label-col", default=None, help="Column name for labels")
@click.option("--name", default=None, help="Name for this dataset")
@click.pass_context
def dataset_custom(ctx, file_path, text_col, label_col, name):
    """Load and cache a custom dataset (CSV/JSON/JSONL/Excel/Parquet)."""
    from rich.console import Console
    from src.datasets import DatasetManager, DatasetStatistics

    config = ctx.obj["config"]
    dm = DatasetManager(config)
    console = Console()
    console.print(f"Loading custom dataset: [cyan]{file_path}[/cyan]")
    data = dm.load_custom(file_path, text_col, label_col, dataset_name=name)
    console.print(f"[green][OK] Loaded {sum(len(v) for v in data.values())} samples[/green]")
    stats = DatasetStatistics(data, {})
    stats.compute_all()
    stats.print_summary()


# ─────────────────────────────────────────────────────────────────────
# encode commands
# ─────────────────────────────────────────────────────────────────────

@cli.group()
def encode():
    """Spike encoding commands."""
    pass


@encode.command("demo")
@click.option("--text", default="The court ruled that the defendant violated Article 6 of the European Convention.", help="Text to encode")
@click.pass_context
def encode_demo(ctx, text):
    """Demonstrate all spike encodings on a sample text."""
    from rich.console import Console
    from rich.table import Table
    import numpy as np
    from src.encoding import ENCODERS

    config = ctx.obj["config"]
    console = Console()
    time_steps = config.get("encoding", {}).get("time_steps", 50)

    # Fake embedding for demo (in production this comes from a transformer)
    rng = np.random.default_rng(42)
    embedding = rng.normal(0, 1, (1, 64)).astype(np.float32)

    console.print(f"\n[bold]Input text:[/bold] {text}")
    console.print(f"[bold]Embedding dim:[/bold] 64 (demo), [bold]Time steps:[/bold] {time_steps}\n")

    table = Table(title="Spike Encoding Summary", show_header=True)
    table.add_column("Encoding", style="bold cyan")
    table.add_column("Spike Shape")
    table.add_column("Total Spikes", justify="right")
    table.add_column("Sparsity", justify="right")
    table.add_column("Avg Firing Rate", justify="right")

    for enc_name, EncoderClass in ENCODERS.items():
        enc = EncoderClass(time_steps=time_steps)
        spk = enc.encode(embedding)
        total = int(spk.sum())
        sparsity = float(1.0 - spk.mean())
        rate = float(spk.mean())
        table.add_row(
            enc_name,
            str(spk.shape),
            str(total),
            f"{sparsity:.2%}",
            f"{rate:.4f}",
        )

    console.print(table)


@encode.command("compare")
@click.option("--dataset", default="case_hold", help="Dataset to use for encoding comparison")
@click.option("--n-samples", default=50, help="Number of samples to encode")
@click.pass_context
def encode_compare(ctx, dataset, n_samples):
    """Compare spike encodings on real transformer embeddings."""
    from rich.console import Console
    from src.datasets import DatasetManager
    from src.models import TransformerBaseline
    from src.encoding import ENCODERS
    from src.visualization import ResearchPlotter
    import numpy as np

    config = ctx.obj["config"]
    console = Console()

    console.print(f"Loading [cyan]{dataset}[/cyan]…")
    dm = DatasetManager(config)
    data = dm.load(dataset, max_samples=n_samples)
    rows = data.get("train", [])[:n_samples]

    console.print("Extracting embeddings…")
    model_key = next(
        (k for k, v in config["models"]["transformers"].items() if v.get("active")),
        "legal_bert",
    )
    model = TransformerBaseline(config, model_key)
    embeddings = model.get_embeddings(rows, batch_size=16)

    console.print("Generating spike trains for all encodings…")
    time_steps = config["encoding"]["time_steps"]
    spike_trains = {}
    for enc_name, EncoderClass in ENCODERS.items():
        enc = EncoderClass(time_steps=time_steps)
        spike_trains[enc_name] = enc.encode(embeddings)
        console.print(f"  [{enc_name}] sparsity={enc.sparsity(spike_trains[enc_name]):.2%}")

    console.print("\nGenerating visualizations…")
    plotter = ResearchPlotter(config)
    raster_path = plotter.plot_encoding_comparison_rasters(spike_trains)
    rates_path = plotter.plot_firing_rates(spike_trains)

    console.print(f"\n[green][OK] Figures saved:[/green]")
    console.print(f"  Rasters: {raster_path}")
    console.print(f"  Firing rates: {rates_path}")


# ─────────────────────────────────────────────────────────────────────
# report command
# ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--results", default=None, help="Path to results JSON file")
@click.option("--format", "fmt", default=None, help="Output format: html|markdown|latex")
@click.pass_context
def report(ctx, results, fmt):
    """Regenerate report from saved results JSON."""
    import json
    from rich.console import Console
    from src.reporting import ReportGenerator

    config = ctx.obj["config"]
    if fmt:
        config["reporting"]["format"] = fmt

    console = Console()

    if results is None:
        # Find most recent results file
        result_files = sorted(Path("storage/results").glob("results_*.json"))
        if not result_files:
            console.print("[red]No results files found. Run 'python main.py run' first.[/red]")
            return
        results = str(result_files[-1])
        console.print(f"Using most recent results: {results}")

    with open(results) as f:
        data = json.load(f)

    gen = ReportGenerator(config)
    path = gen.generate(
        title="A Semantic- and Energy-Aware Study of Spike Encoding for Legal Text Classification",
        experiment_config=config,
        classification_results=data.get("transformer"),
        semantic_results=data.get("semantic"),
        energy_results=data.get("energy"),
        domain_shift_results=data.get("domain_shift"),
    )
    console.print(f"[green][OK] Report generated → {path}[/green]")


# ─────────────────────────────────────────────────────────────────────
# info command
# ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.pass_context
def info(ctx):
    """Display framework info and current configuration summary."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    config = ctx.obj["config"]
    console = Console()

    console.print(Panel.fit(
        "[bold cyan]Spike-Legal-NLP Research Framework v1.0[/bold cyan]\n\n"
        "[bold]Research:[/bold] A Semantic- and Energy-Aware Study of Spike Encoding\n"
        "          for Legal Text Classification Under Domain Shift\n\n"
        "[bold]Hypotheses:[/bold]\n"
        "  H1: Spike encoding achieves competitive classification performance\n"
        "  H2: Time-based encodings preserve legal semantics better than rate coding\n"
        "  H3: Energy improvements exist but shrink with memory access included\n"
        "  H4: Domain shift affects all models differently",
        border_style="blue",
    ))

    # Active datasets
    ds_table = Table(title="Active Datasets", show_header=True)
    ds_table.add_column("Key")
    ds_table.add_column("Task")
    ds_table.add_column("Active")
    for k, v in config.get("datasets", {}).get("available", {}).items():
        active = "[green][OK][/green]" if v.get("active") else "[red][X][/red]"
        ds_table.add_row(k, v.get("task", "—"), active)
    console.print(ds_table)

    # Active models
    m_table = Table(title="Transformer Models", show_header=True)
    m_table.add_column("Key")
    m_table.add_column("Model")
    m_table.add_column("Active")
    for k, v in config.get("models", {}).get("transformers", {}).items():
        active = "[green][OK][/green]" if v.get("active") else "[red][X][/red]"
        m_table.add_row(k, v.get("name", "—"), active)
    console.print(m_table)

    # Encodings
    e_table = Table(title="Spike Encodings", show_header=True)
    e_table.add_column("Method")
    e_table.add_column("Enabled")
    for k, v in config.get("encoding", {}).get("methods", {}).items():
        enabled = "[green][OK][/green]" if v.get("enabled") else "[red][X][/red]"
        e_table.add_row(k, enabled)
    console.print(e_table)

    console.print(f"\n[bold]Time Steps:[/bold] {config.get('encoding', {}).get('time_steps', 50)}")
    console.print(f"[bold]Output directory:[/bold] storage/results/")
    console.print(f"\n[bold]Quick start:[/bold]")
    console.print("  python main.py run --quick               # Fast test run")
    console.print("  python main.py run --dataset case_hold   # Full run on CaseHOLD")
    console.print("  python main.py dataset list              # List datasets")
    console.print("  python main.py encode demo               # Demo spike encoding")


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Default: show info if no arguments
    if len(sys.argv) == 1:
        sys.argv.append("info")
    cli(obj={})
