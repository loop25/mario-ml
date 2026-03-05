"""Quick test to verify the environment works after NumPy 2.0 fix."""
import warnings
warnings.filterwarnings('ignore')

print("Testing environment creation...")
from gym_super_mario_bros import SuperMarioBrosEnv
env = SuperMarioBrosEnv()
print("  SuperMarioBrosEnv created OK")

obs = env.reset()
print(f"  Reset OK - observation shape: {obs.shape}")

obs, reward, done, info = env.step(0)
print(f"  Step OK - reward: {reward}, done: {done}")

env.close()
print("  Close OK")
print("\nAll tests passed! Environment is working.")
