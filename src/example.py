import asyncio
import os
from seeact.agent import SeeActAgent

# Setup your API Key here, or pass through environment
# os.environ["OPENAI_API_KEY"] = "Your API KEY Here"
# os.environ["GEMINI_API_KEY"] = "Your API KEY Here"

async def run_agent():
    agent = None
    try:
        agent = SeeActAgent(model="gpt-5", task_id="example_task")
        await agent.start()
        while not agent.complete_flag:
            prediction_dict = await agent.predict()
            await agent.execute(prediction_dict)
    except Exception as e:
        print(f"Error during agent execution: {e}")
    finally:
        if agent:
            try:
                await agent.stop()
            except Exception as stop_e:
                print(f"Error stopping agent: {stop_e}")
                # Try emergency save if available
                try:
                    if hasattr(agent, '_emergency_save'):
                        emergency_file = agent._emergency_save(f"Agent stop failed in example: {stop_e}")
                        if emergency_file:
                            print(f"Emergency save completed: {emergency_file}")
                        else:
                            print("Emergency save also failed")
                except Exception as emergency_e:
                    print(f"CRITICAL: Emergency save failed: {emergency_e}")

if __name__ == "__main__":
    asyncio.run(run_agent())
