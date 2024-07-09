import llama_index.core
import phoenix as px

class Observability:
    """
    Observability class for the LlamaIndex app.
    """
    def __init__(self):
        """
        Initialize the Observability class.
        """
        self.enabled = False
        
    def enable(self):
        """
        Enable observability.
        """
        if not self.enabled:
            try:
                px.launch_app()
                llama_index.core.set_global_handler("arize_phoenix")
                self.enabled = True
            except Exception as e:
                print(f"Failed to enable observability: {e}")
                
    def disable(self):
        """
        Disable observability.
        """
        if self.enabled:
            try:
                px.close_app()
                self.enabled = False
            except Exception as e:
                print(f"Failed to disable observability: {e}")
        
    