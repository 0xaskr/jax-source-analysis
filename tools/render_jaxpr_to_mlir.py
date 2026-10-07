#!/usr/bin/env python3
"""Compatibility entrypoint for the consolidated JAX/Jaxpr architecture SVG."""


def main():
    from render_software_stack_component_flows import main as render_selected
    render_selected(only='jax')


if __name__ == '__main__':
    main()
