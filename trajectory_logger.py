import json
import time
from guardrails import output_guardrail
class TrajectoryLogger:
    """Логування траєкторії виконання агента."""
    _instance = None
 
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(TrajectoryLogger, cls).__new__(cls, *args, **kwargs)
            cls._instance.steps = []
            cls._instance.start_time = time.monotonic()
        return cls._instance
 
    def log_step(self, agent_name: str, step_num: int, node: str,
                 input_data: str, output_data: str,
                 tool_calls: list = None):
        
        safe_input, _ = output_guardrail(str(input_data))
        safe_output, _ = output_guardrail(str(input_data))
        
        self.steps.append({
            'agent_name': agent_name,
            'step': step_num,
            'node': node,
            'input': safe_input,
            'output': safe_output,
            'tool_calls': tool_calls or [],
        })
 
    def save(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'total_steps': len(self.steps),
                'total_time_ms': int((time.monotonic() - self.start_time) * 1000),
                'trajectory': self.steps,
            }, f, ensure_ascii=False, indent=2)
        print(f'Trajectory saved: {filepath} ({len(self.steps)} steps)')

logger = TrajectoryLogger()