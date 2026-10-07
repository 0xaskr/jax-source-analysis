#!/usr/bin/env python3
"""Compatibility entrypoint for the selected overview-style component SVGs."""


def main():
    from render_software_stack_component_flows import main as render_selected
    render_selected()


if __name__ == '__main__':
    main()
