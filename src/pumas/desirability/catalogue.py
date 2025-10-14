from pumas.architecture.catalogue import Catalogue
from pumas.desirability.base_models import Desirability
from pumas.desirability.bell import Bell
from pumas.desirability.double_sigmoid import DoubleSigmoid
from pumas.desirability.exponential_decay import ExponentialDecay
from pumas.desirability.multistep import MultiStep
from pumas.desirability.sigmoid import Sigmoid
from pumas.desirability.sigmoid_bell import SigmoidBell
from pumas.desirability.step import LeftStep, RightStep, Step
from pumas.desirability.value_mapping import ValueMapping

desirability_catalogue = Catalogue(item_type=Desirability)

# Register all desirabilities here
desirability_catalogue.register(name=Sigmoid.name, item=Sigmoid)

desirability_catalogue.register(name=DoubleSigmoid.name, item=DoubleSigmoid)

desirability_catalogue.register(name=Bell.name, item=Bell)

desirability_catalogue.register(name=SigmoidBell.name, item=SigmoidBell)

desirability_catalogue.register(name=MultiStep.name, item=MultiStep)

desirability_catalogue.register(name=LeftStep.name, item=LeftStep)

desirability_catalogue.register(name=RightStep.name, item=RightStep)

desirability_catalogue.register(name=Step.name, item=Step)

desirability_catalogue.register(name=ValueMapping.name, item=ValueMapping)

desirability_catalogue.register(name=ExponentialDecay.name, item=ExponentialDecay)

# Add more registrations as new desirabilities are created
# desirability_catalogue.register(name=<name>", item=<NewDesirability>)
