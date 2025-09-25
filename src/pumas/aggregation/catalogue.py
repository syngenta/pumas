from pumas.aggregation.base_models import Aggregation
from pumas.aggregation.weighted_arithmetic_mean import WeightedArithmeticMeanAggregation
from pumas.aggregation.weighted_deviation_index import WeightedDeviationIndexAggregation
from pumas.aggregation.weighted_geometric_mean import WeightedGeometricMeanAggregation
from pumas.aggregation.weighted_harmonic_mean import WeightedHarmonicMeanAggregation
from pumas.aggregation.weighted_product import WeightedProductAggregation
from pumas.aggregation.weighted_summation import WeightedSummationAggregation
from pumas.architecture.catalogue import Catalogue

aggregation_catalogue = Catalogue(item_type=Aggregation)


# Register all aggregation methods here
aggregation_catalogue.register(
    name=WeightedArithmeticMeanAggregation.name, item=WeightedArithmeticMeanAggregation
)

aggregation_catalogue.register(
    name=WeightedGeometricMeanAggregation.name, item=WeightedGeometricMeanAggregation
)

aggregation_catalogue.register(
    name=WeightedHarmonicMeanAggregation.name, item=WeightedHarmonicMeanAggregation
)
aggregation_catalogue.register(
    name=WeightedDeviationIndexAggregation.name, item=WeightedDeviationIndexAggregation
)

aggregation_catalogue.register(
    name=WeightedSummationAggregation.name, item=WeightedSummationAggregation
)

aggregation_catalogue.register(
    name=WeightedProductAggregation.name, item=WeightedProductAggregation
)

# register new aggregation methods here
# aggregation_catalogue.register(name=<name>", item=<NewAggregation>)
