"""Command-line interface for simpli-agent."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import typer
    TYPER_AVAILABLE = True
except ImportError:
    TYPER_AVAILABLE = False

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.syntax import Syntax
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from .config import AgentConfig, load_config, create_agent_from_config
from .core import Agent


def _get_console():
    if RICH_AVAILABLE:
        return Console()
    return None


def _print(text: str, style: str | None = None) -> None:
    console = _get_console()
    if console and style:
        console.print(text, style=style)
    else:
        print(text)


def _print_markdown(text: str) -> None:
    console = _get_console()
    if console:
        console.print(Markdown(text))
    else:
        print(text)


def _print_code(code: str, lang: str = "python") -> None:
    console = _get_console()
    if console:
        console.print(Syntax(code, lang, theme="monokai"))
    else:
        print(code)


app = None
if TYPER_AVAILABLE:
    app = typer.Typer(
        name="simpli-agent",
        help="Embedded Python agent runtime",
        no_args_is_help=True,
    )


def run_chat(agent: Agent, streaming: bool = False) -> None:
    """Run interactive chat loop."""
    _print_markdown("**simpli-agent chat** (type 'exit' or 'quit' to leave)")

    while True:
        try:
            user_input = input("\n> ").strip()
            if user_input.lower() in ("exit", "quit", "q"):
                _print("Goodbye!", "green")
                break
            if not user_input:
                continue

            if streaming:
                _print("Assistant: ", style="bold blue", end="")
                for chunk in agent.stream(user_input):
                    _print(chunk + " ", style="blue", end="", flush=True)
                _print()
            else:
                response = agent.run(user_input)
                _print_markdown(f"**Assistant:** {response}")

        except KeyboardInterrupt:
            _print("\nInterrupted", "yellow")
            break
        except EOFError:
            break
        except Exception as e:
            _print(f"Error: {e}", "red")


def run_single(agent: Agent, prompt: str, streaming: bool = False) -> None:
    """Run a single prompt."""
    if streaming:
        for chunk in agent.stream(prompt):
            _print(chunk + " ", style="blue", end="", flush=True)
        _print()
    else:
        response = agent.run(prompt)
        _print_markdown(response)


if TYPER_AVAILABLE:
    @app.command()
    def chat(
        config: str | None = typer.Option(None, "-c", "--config", help="Config file (TOML/YAML)"),
        model: str = typer.Option("gpt-4o", "-m", "--model", help="Model name"),
        backend: str = typer.Option("hermes", "-b", "--backend", help="Backend (hermes, openai, anthropic, ollama)"),
        db: str = typer.Option(":memory:", "--db", help="Database path"),
        system: str | None = typer.Option(None, "--system", help="System prompt"),
        streaming: bool = typer.Option(False, "--stream", help="Stream responses"),
    ):
        """Start an interactive chat session."""
        if config:
            cfg = load_config(config)
            agent = create_agent_from_config(cfg)
        else:
            from .backends import HermesBackend, OpenAIBackend, AnthropicBackend, OllamaBackend

            if backend == "openai":
                b = OpenAIBackend()
            elif backend == "anthropic":
                b = AnthropicBackend()
            elif backend == "ollama":
                b = OllamaBackend()
            else:
                b = HermesBackend()

            agent = Agent(
                model=model,
                backend=b,
                db_path=db,
                system_prompt=system,
            )

        run_chat(agent, streaming)

    @app.command()
    def run(
        prompt: str = typer.Argument(..., help="Prompt to run"),
        config: str | None = typer.Option(None, "-c", "--config", help="Config file (TOML/YAML)"),
        model: str = typer.Option("gpt-4o", "-m", "--model", help="Model name"),
        backend: str = typer.Option("hermes", "-b", "--backend", help="Backend (hermes, openai, anthropic, ollama)"),
        db: str = typer.Option(":memory:", "--db", help="Database path"),
        system: str | None = typer.Option(None, "--system", help="System prompt"),
        streaming: bool = typer.Option(False, "--stream", help="Stream responses"),
        output: str | None = typer.Option(None, "-o", "--output", help="Output format (text, json, markdown)"),
    ):
        """Run a single prompt."""
        if config:
            cfg = load_config(config)
            agent = create_agent_from_config(cfg)
        else:
            from .backends import HermesBackend, OpenAIBackend, AnthropicBackend, OllamaBackend

            if backend == "openai":
                b = OpenAIBackend()
            elif backend == "anthropic":
                b = AnthropicBackend()
            elif backend == "ollama":
                b = OllamaBackend()
            else:
                b = HermesBackend()

            agent = Agent(
                model=model,
                backend=b,
                db_path=db,
                system_prompt=system,
            )

        run_single(agent, prompt, streaming)

    @app.command()
    def tools(
        config: str | None = typer.Option(None, "-c", "--config", help="Config file (TOML/YAML)"),
    ):
        """List available tools from config."""
        if config:
            cfg = load_config(config)
            agent = create_agent_from_config(cfg)
            _print("Registered tools:", "bold")
            for name in agent.tools:
                _print(f"  - {name}")
        else:
            _print("No config provided. Use --config to load a configuration with tools.", "yellow")

    @app.command()
    def serve(
        config: str = typer.Option(..., "-c", "--config", help="Config file (TOML/YAML)"),
        host: str = typer.Option("127.0.0.1", "--host", help="Host to bind"),
        port: int = typer.Option(8000, "--port", help="Port to bind"),
    ):
        """Start a FastAPI server for the agent."""
        try:
            import uvicorn
            from fastapi import FastAPI, HTTPException
            from pydantic import BaseModel
        except ImportError:
            _print("FastAPI and uvicorn required. Install with: pip install fastapi uvicorn", "red")
            sys.exit(1)

        cfg = load_config(config)
        agent = create_agent_from_config(cfg)

        app = FastAPI(title="simpli-agent API")

        class RunRequest(BaseModel):
            prompt: str
            streaming: bool = False
            output_model: str | None = None

        class RunResponse(BaseModel):
            response: str

        @app.post("/run", response_model=RunResponse)
        async def run_endpoint(req: RunRequest):
            try:
                result = agent.run(req.prompt)
                return RunResponse(response=str(result))
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        _print(f"Starting server on {host}:{port}", "green")
        uvicorn.run(app, host=host, port=port)


def main():
    """Entry point for CLI."""
    if not TYPER_AVAILABLE:
        _print("CLI requires typer. Install with: pip install typer", "red")
        sys.exit(1)

    if app:
        app()


if __name__ == "__main__":
    main()