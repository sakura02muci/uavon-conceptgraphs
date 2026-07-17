"""Configuration loader for ConceptGraphs UAV-ON."""
import os
from pathlib import Path
from typing import Optional


def load_env_file(env_path: Optional[str] = None) -> None:
    """Load environment variables from .env file."""
    if env_path is None:
        # Try to find .env file in project root
        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parent.parent  # Go up to UAV_ON root
        env_path = project_root / ".env"
    else:
        env_path = Path(env_path)
    
    if not env_path.exists():
        return
    
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            # Parse KEY=VALUE
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Don't override existing environment variables
                if key not in os.environ:
                    os.environ[key] = value


def get_deepseek_key() -> Optional[str]:
    """Get DeepSeek API key from environment or .env file."""
    # Try to load from .env first
    load_env_file()
    
    # Return key from environment
    return os.environ.get('DEEPSEEK_API_KEY')


def get_openai_key() -> Optional[str]:
    """Get OpenAI API key from environment or .env file."""
    load_env_file()
    return os.environ.get('OPENAI_API_KEY')


# Auto-load when module is imported
load_env_file()
