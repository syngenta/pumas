from pumas.architecture.catalogue import Catalogue
from pumas.framework.base_models import BaseFramework
from pumas.framework.score_value import FrameworkScoreValue

framework_catalogue = Catalogue(item_type=BaseFramework)

# Register all frameworks here
framework_catalogue.register(name="score_value", item=FrameworkScoreValue)

# Add more registrations as new frameworks are created
# framework_catalogue.register(name=<name>", item=<NewFramework>)
