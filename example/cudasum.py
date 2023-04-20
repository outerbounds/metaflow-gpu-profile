from metaflow import (
    Parameter,
    parallel_map,
    conda,
    resources,
    FlowSpec,
    step,
)

from gpu_profile import gpu_profile


class CudaSumFlow(FlowSpec):
    num_iter = Parameter("num_iter", default=20, help="number of iterations per task")
    num_tasks = Parameter("num_tasks", default=1, help="number of parallel GPU tasks")

    @step
    def start(self):
        self.indices = list(range(self.num_tasks))
        self.next(self.cudapush, foreach="indices")

    @resources(memory=20000, gpu=2)
    @conda(
        python="3.9.13",
        libraries={"numba": "0.56.4", "cudatoolkit": "10.2.89", "matplotlib": "3.7.0"},
    )
    @gpu_profile(interval=1)
    @step
    def cudapush(self):
        import cudatest
        import time
        from random import seed, randint

        seed(self.index)
        for i in range(self.num_iter):
            params = [
                {
                    "array_size": randint(10_000_000, 1_000_000_000),
                    "num_iter": randint(10, 1000),
                    "device": i,
                }
                for i in range(self.gpu_profile_num_gpus)
            ]
            parallel_map(lambda x: cudatest.push_cuda(**x), params)
            print(f"iteration {i} done")
        self.next(self.join)

    @step
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    CudaSumFlow()
