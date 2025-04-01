from pumas.architecture.catalogue import Catalogue
from pumas.desirability.base_models import Desirability
from pumas.desirability.bell import Bell
from pumas.desirability.double_sigmoid import DoubleSigmoid
from pumas.desirability.multistep import MultiStep
from pumas.desirability.sigmoid import Sigmoid
from pumas.desirability.sigmoid_bell import SigmoidBell
from pumas.desirability.step import LeftStep, RightStep, Step
from pumas.desirability.value_mapping import ValueMapping

desirability_catalogue = Catalogue(item_type=Desirability)

# Register all desirabilities here
desirability_catalogue.register(name="sigmoid", item=Sigmoid)

desirability_catalogue.register(name="double_sigmoid", item=DoubleSigmoid)

desirability_catalogue.register(name="bell", item=Bell)

desirability_catalogue.register(name="sigmoid_bell", item=SigmoidBell)

desirability_catalogue.register(name="multistep", item=MultiStep)

desirability_catalogue.register(name="leftstep", item=LeftStep)

desirability_catalogue.register(name="rightstep", item=RightStep)

desirability_catalogue.register(name="step", item=Step)

desirability_catalogue.register(name="value_mapping", item=ValueMapping)

# Add more registrations as new desirabilities are created
# desirability_catalogue.register(name=<name>", item=<NewDesirability>)
