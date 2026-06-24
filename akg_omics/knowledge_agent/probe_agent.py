import argparse
import json

try:
    # Package mode: python -m akg_omics.knowledge_agent.probe_agent
    from .agent import merge_registry_and_discovery, run_discovery_agent
    from .registry import load_registry
    from .selector import select_sources
    from .task_schema import TaskSpec
except ImportError:
    # Script mode fallback
    from knowledge_agent.agent import merge_registry_and_discovery, run_discovery_agent
    from knowledge_agent.registry import load_registry
    from knowledge_agent.selector import select_sources
    from knowledge_agent.task_schema import TaskSpec


def main():
    parser = argparse.ArgumentParser(description="Probe knowledge selection agent.")
    parser.add_argument("--task-id", default="probe")
    parser.add_argument("--source", default="he,protein", help="comma-separated source modalities")
    parser.add_argument("--target", default="gene")
    parser.add_argument("--max-sources", type=int, default=4)
    parser.add_argument("--registry", default=None)
    args = parser.parse_args()

    src = [x.strip() for x in args.source.split(",") if x.strip()]
    task = TaskSpec(
        task_id=args.task_id,
        source_modalities=src,
        target_modality=args.target,
    )
    registry = load_registry(args.registry)
    discovery = run_discovery_agent(task=task, registry_sources=registry)
    merged = merge_registry_and_discovery(registry, discovery)
    result = select_sources(task=task, registry_sources=merged, max_sources=args.max_sources)
    payload = {
        "task": task.to_dict(),
        "discovery": discovery,
        "selection": result.to_dict(),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
