from .statistical import StatisticalCritic
from .methodology import MethodologyCritic
from .ml import MLCritic
from .microstructure import MicrostructureCritic

CRITIC_REGISTRY: dict[str, type] = {
    "statistical": StatisticalCritic,
    "methodology": MethodologyCritic,
    "ml": MLCritic,
    "microstructure": MicrostructureCritic,
}
