from pumas.aggregation.base_models import BaseAggregation
from pumas.aggregation.weighted_arithmetic_mean import WeightedArithmeticMeanAggregation
from pumas.aggregation.weighted_deviation_index import WeightedDeviationIndexAggregation
from pumas.aggregation.weighted_geometric_mean import WeightedGeometricMeanAggregation
from pumas.aggregation.weighted_harmonic_mean import WeightedHarmonicMeanAggregation
from pumas.aggregation.weighted_product import WeightedProductAggregation
from pumas.aggregation.weighted_summation import WeightedSummationAggregation
from pumas.architecture.catalogue import Catalogue

aggregation_catalogue = Catalogue(item_type=BaseAggregation)


# Register all aggregation methods here
aggregation_catalogue.register(
    name="arithmetic_mean", item=WeightedArithmeticMeanAggregation
)

aggregation_catalogue.register(
    name="geometric_mean", item=WeightedGeometricMeanAggregation
)

aggregation_catalogue.register(
    name="harmonic_mean", item=WeightedHarmonicMeanAggregation
)
aggregation_catalogue.register(
    name="deviation_index", item=WeightedDeviationIndexAggregation
)

aggregation_catalogue.register(name="summation", item=WeightedSummationAggregation)

aggregation_catalogue.register(name="product", item=WeightedProductAggregation)

# register new aggregation methods here
# aggregation_catalogue.register(name=<name>", item=<NewAggregation>)
