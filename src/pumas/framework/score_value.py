from pumas.dataframes.dataframe import DataFrame
from pumas.dataframes.dataframe_utils import concat
from pumas.framework.base_models import BaseFramework


class FrameworkScoreValue(BaseFramework):
    def compute(self):
        self.desirability_dataframe = self._build_desirability_dataframe()
        self.aggregated_dataframe = self._build_aggregated_scores_dataframe()

    def _build_desirability_dataframe(self):
        # crete a dataframe for each objective

        # create an individual dataframe for each objective
        # by applying the desirability function to the property dataframe column
        desirability_score_dataframe_list = [
            self._properties_dataframe.apply_elementwise_column(
                func=v.compute_numeric, column_name=k, new_column_name=k
            )
            for k, v in self._desirability_functions_map.items()
        ]
        # concatenate the individual dataframes into a single dataframe
        # the index is the same for all dataframes,
        # we concatenate them along the columns (axis=1)
        # getting the union of the columns (join="outer")

        desirability_dataframe = concat(
            dataframes=desirability_score_dataframe_list, axis=1, join="outer"
        )
        return desirability_dataframe

    def _build_aggregated_scores_dataframe(self):
        keys = [objective.name for objective in self._scoring_profile.objectives]
        weights = [objective.weight for objective in self._scoring_profile.objectives]

        aggregation_function = self._aggregation_catalogue.get(
            name=self._scoring_profile.aggregation_function.name
        )()
        aggregated_scores = []
        index_values = []
        for i, row in enumerate(self.desirability_dataframe.row_data):
            index = self.desirability_dataframe.index.values[i]
            values = [row[k] for k in keys]
            aggregated_score = aggregation_function.compute_numeric(
                values=values, weights=weights
            )
            index_values.append(index)
            aggregated_scores.append(aggregated_score)

        aggregated_scores_dataframe = DataFrame(
            row_data=[
                {"aggregated_score": aggregated_score}
                for aggregated_score in aggregated_scores
            ],
            index=index_values,
        )
        return aggregated_scores_dataframe
