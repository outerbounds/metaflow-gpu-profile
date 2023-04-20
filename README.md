
# GPU profiling card for Metaflow

Use this [custom Metaflow card](https://docs.metaflow.org/metaflow/visualizing-results/easy-custom-reports-with-card-components) to observe GPU utilization and memory consumption on any Metaflow task.

## Requirements

The uses the `nvidia-smi` command-line tool to retrieve the metrics. Make sure the tool is installed on the image
you use. Most CUDA-enabled images include it by default.

To see pretty charts, ensure that `matplotlib` is installed either in your `@conda` environment or in the image.

## Usage

1. Copy `gpu_profile.py` in the same directory with your flow file.

2. Add `from gpu_profile import gpu_profile` at the top of the flow file.

3. Include `@gpu_profile()` (remember the parentheses) above every step that you want to monitor.

Run the flow as usual and observe the card, after a task has finished, either through [command-line,
notebook, or Metaflow UI](https://docs.metaflow.org/metaflow/visualizing-results/effortless-task-inspection-with-default-cards#cards-are-stored-and-versioned-automatically).

## Example

See `example/cudasum.py` for an example that exercises GPU and monitors utilization with this card.
