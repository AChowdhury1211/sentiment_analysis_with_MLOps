

import mlflow
from typing import Protocol
from dataclasses import dataclass

class ExperimentTracker(Protocol):
    
    def __start__(self):
        ...

    def log(self):
        ...

    def end(self):
        ...

@dataclass
class MLFlowTracker:
   

    experiment_name: str
    tracking_uri: str
    run_name: str
    experiment: bool
    
    def __start__(self) -> None:
        
        try:
            mlflow.set_tracking_uri(self.tracking_uri)

        except ConnectionError:
            print(f"Cannot connect to {self.tracking_uri}. Please check and validate the URI!!")

        else:
            if self.experiment:
                exp_id = mlflow.create_experiment(self.experiment_name)
                experiment = mlflow.get_experiment(exp_id)

            else:
                experiment = mlflow.set_experiment(self.experiment_name)

            mlflow.start_run(run_name = self.run_name,
                            experiment_id = experiment.experiment_id)
    
    def log(self) -> None:
       
        self.__start__()
        mlflow.tensorflow.autolog()

    def end(self) -> None:
        
        mlflow.end_run()