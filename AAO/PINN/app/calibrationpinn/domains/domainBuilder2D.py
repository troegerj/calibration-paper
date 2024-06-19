# Standard library imports

# Third-party imports

# Local library imports
from calibrationpinn.domains.dataselection import DataSelectionFuncProtocol
from calibrationpinn.domains.domain2D import (
    Domain2DWithSolution,
    Domain2DWithoutSolution,
)
from calibrationpinn.domains.inputreader import InputDataReader2DProtocol
from calibrationpinn.domains.splitDataFunc import split_in_train_and_valid_data
from calibrationpinn.typeAliases import PRNGKey


class DomainBuilder2D:
    def build_domain_with_solution(
        self,
        input_reader: InputDataReader2DProtocol,
        data_selection_func: DataSelectionFuncProtocol,
        PRNG_key: PRNGKey,
    ) -> Domain2DWithSolution:
        observation_data = input_reader.read_observation_data()
        force_data = input_reader.read_force_data()
        solution_data = input_reader.read_solution_data()
        return Domain2DWithSolution(
            observation_data=observation_data,
            force_data=force_data,
            solution_data=solution_data,
            data_selection_func=data_selection_func,
            PRNG_key=PRNG_key,
        )

    def build_domain_without_solution(
        self,
        input_reader: InputDataReader2DProtocol,
        proportion_training_data: float,
        data_selection_func: DataSelectionFuncProtocol,
        PRNG_key: PRNGKey,
    ) -> Domain2DWithoutSolution:
        observation_data = input_reader.read_observation_data()
        force_data = input_reader.read_force_data()
        return Domain2DWithoutSolution(
            observation_data=observation_data,
            force_data=force_data,
            proportion_train_data=proportion_training_data,
            split_data_func=split_in_train_and_valid_data,
            data_selection_func=data_selection_func,
            PRNG_key=PRNG_key,
        )
