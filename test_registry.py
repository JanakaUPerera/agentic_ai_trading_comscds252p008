from src.agent_tools import TOOL_REGISTRY, list_tool_names, get_tool_descriptions

print("Registered tools:")
for name in list_tool_names():
    print("-", name)

print("\nTool descriptions:")
print(get_tool_descriptions())