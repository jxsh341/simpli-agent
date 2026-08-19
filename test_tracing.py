from simpli_agent import Agent, Tracer, CallbackHandler

# Test basic tracer
tracer = Tracer("test")

with tracer.span("test_span") as span:
    span.set_attribute("key", "value")
    span.add_event("test_event", {"data": "test"})

print("Spans:", len(tracer.get_spans()))
print("Span:", tracer.get_spans()[0].name, tracer.get_spans()[0].attributes)

# Test callback handler
callback_handler = CallbackHandler()
events = []

callback_handler.on_run_start(lambda run_id, data: events.append(("run_start", run_id, data)))
callback_handler.on_run_end(lambda run_id, result, duration: events.append(("run_end", run_id, result, duration)))
callback_handler.on_tool_start(lambda name, args: events.append(("tool_start", name, args)))
callback_handler.on_tool_end(lambda name, result, duration: events.append(("tool_end", name, result, duration)))
callback_handler.on_error(lambda e: events.append(("error", str(e))))

agent = Agent(model="test-model", tracer=callback_handler)

@agent.tool
def add(a: int, b: int) -> int:
    return a + b

result = agent.run("Add 2 and 3")
print("Result:", result)
print("Events:", events)