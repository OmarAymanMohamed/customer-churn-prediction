import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Callable

from src.config.config import LOG_FILE
from src.utils.logger import setup_logger, log_error
from src.data.processor import DataProcessor
from src.models.trainer import ModelTrainer

logger = setup_logger(__name__)

class PipelineRunner:
    def __init__(self):
        self.steps = [
            self.setup_environment,
            self.preprocess_data,
            self.train_models,
            self.deploy_application
        ]
        
    def setup_environment(self) -> bool:
        """Set up the Python environment and install dependencies."""
        try:
            logger.info("Setting up environment...")
            
            # Create necessary directories
            for directory in ['data', 'processed_data', 'model_artifacts', 'logs']:
                Path(directory).mkdir(exist_ok=True)
            
            # Install dependencies
            os.system(f"{sys.executable} -m pip install -r requirements.txt")
            
            logger.info("Environment setup completed successfully")
            return True
            
        except Exception as e:
            log_error(logger, e, "Error setting up environment")
            return False
            
    def preprocess_data(self) -> bool:
        """Run the data preprocessing step."""
        try:
            logger.info("Starting data preprocessing...")
            
            # Initialize and run processor
            processor = DataProcessor()
            processor.main()
            
            logger.info("Data preprocessing completed successfully")
            return True
            
        except Exception as e:
            log_error(logger, e, "Error in data preprocessing")
            return False
            
    def train_models(self) -> bool:
        """Run the model training step."""
        try:
            logger.info("Starting model training...")
            
            # Initialize and run trainer
            trainer = ModelTrainer()
            trainer.main()
            
            logger.info("Model training completed successfully")
            return True
            
        except Exception as e:
            log_error(logger, e, "Error in model training")
            return False
            
    def deploy_application(self) -> bool:
        """Deploy the Streamlit application."""
        try:
            logger.info("Starting application deployment...")
            
            # Run the Streamlit app
            os.system(f"{sys.executable} -m streamlit run app.py")
            
            logger.info("Application deployed successfully")
            return True
            
        except Exception as e:
            log_error(logger, e, "Error deploying application")
            return False
            
    def run_pipeline(self) -> bool:
        """Run the entire pipeline."""
        logger.info("Starting pipeline execution...")
        
        for step in self.steps:
            logger.info(f"Running step: {step.__name__}")
            
            if not step():
                logger.error(f"Pipeline failed at step: {step.__name__}")
                return False
                
            logger.info(f"Step completed: {step.__name__}")
            time.sleep(1)  # Add a small delay between steps
            
        logger.info("Pipeline completed successfully")
        return True

def main():
    """Main function to run the pipeline."""
    try:
        # Initialize pipeline runner
        runner = PipelineRunner()
        
        # Run the pipeline
        success = runner.run_pipeline()
        
        if success:
            logger.info("Pipeline completed successfully")
            sys.exit(0)
        else:
            logger.error("Pipeline failed")
            sys.exit(1)
            
    except Exception as e:
        log_error(logger, e, "Error in pipeline execution")
        sys.exit(1)

if __name__ == "__main__":
    main() 