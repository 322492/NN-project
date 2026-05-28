import wandb


def init_wandb(config, enabled=True, project="ena24-baseline", job_type=None):
    if not enabled:
        return None

    run = wandb.init(
        project=project,
        config=config,
        job_type=job_type,
        mode="online",
    )

    print("W&B run:", run.url)

    return run


def wandb_log(run, metrics):
    if run is None:
        return

    run.log(metrics)

    for key, value in metrics.items():
        run.summary[key] = value


def finish_wandb(run):
    if run is not None:
        run.finish()

def log_model_artifact(run, checkpoint_path, artifact_name="baseline_cnn_best"):
    if run is None:
        return

    artifact = wandb.Artifact(
        name=artifact_name,
        type="model"
    )

    artifact.add_file(str(checkpoint_path))

    run.log_artifact(
        artifact,
        aliases=["best"]
    )