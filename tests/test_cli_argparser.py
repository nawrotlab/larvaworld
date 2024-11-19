from cli_test_helpers import shell, ArgvContext


def test_entrypoint():
    """
    Is entrypoint script installed?
    """
    result = shell("larvaworld --help")
    assert result.exit_code == 0
    assert "usage" in result.stdout


# def test_arg_experiment():
#     """Is argument experiment available?"""
#     with ArgvContext(f"{cli_entry_point} Replay --help"):
#         cli.main()
#         args = SimModeParser.parse_args()

#     assert args.experiment == 'Replay'
