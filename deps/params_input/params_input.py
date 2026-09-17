"""Legacy command-line parameter parser used by the training experiments.

The parser intentionally keeps the original dictionary-backed API and option
names.  It is not an ``argparse`` replacement because several historical
scripts rely on the permissive, sequential parsing behaviour.
"""

import sys


def numcheck(input):
    """Return whether ``input`` can be interpreted as an integer or float."""

    try:
        int(input)
        return True
    except ValueError:
        try:
            float(input)
            return True
        except ValueError:
            return False


def _consume_list(argv, start, limit):
    """Read a value followed by all subsequent non-option values."""

    values = [argv[start]]
    index = start + 1

    while index < limit and argv[index][0] != "-":
        values.append(argv[index])
        index += 1

    return values


class QParams:
    """Dictionary-style storage for the original experiment parameters."""

    _STRING_OPTIONS = {
        "-pa_im": "pa_im",
        "-ft_im": "ft_im",
        "-ft_gt": "ft_gt",
        "-pa_gt": "pa_gt",
        "-pa_fg": "pa_fg",
        "-pa_net": "pa_net",
        "-pa_out": "pa_out",
        "-check_data": "check_data",
        "-gpuid": "gpuid",
        "-version": "version",
        "-idx_net": "idx_net",
    }
    _LIST_OPTIONS = {
        "-list_pa_im": "list_pa_im",
        "-list_pa_fg": "list_pa_fg",
        "-list_pa_gt": "list_pa_gt",
    }
    _FLOAT_OPTIONS = {
        "-bk_rate": "bk_rate",
        "-fg_rate": "fg_rate",
        "-vid_rate": "vid_rate",
        "-check_rate": "check_rate",
    }
    _INT_OPTIONS = {
        "-epochnum": "epochnum",
        "-preload": "preload",
    }

    def setParams(self, argc, argv):
        """Parse up to ``argc`` entries from ``argv`` into ``m_params``."""

        self.m_params["ft_im"] = "jpg"
        self.m_params["ft_gt"] = "png"
        self.m_params["preload"] = -1

        cnt = 0
        while cnt < argc:
            option = argv[cnt]

            if option in self._STRING_OPTIONS:
                self.m_params[self._STRING_OPTIONS[option]] = argv[cnt + 1]

            elif option in self._LIST_OPTIONS:
                self.m_params[self._LIST_OPTIONS[option]] = _consume_list(argv, cnt + 1, argc)

            elif option == "-idx_list":
                self.m_params["idx_list"] = []
                current_indices = []
                index = cnt + 1

                while index < argc and numcheck(argv[index]):
                    value = int(argv[index])
                    if value == -1:
                        self.m_params["idx_list"].append(current_indices)
                        current_indices = []
                    else:
                        current_indices.append(value)
                    index += 1

            elif option == "-imgs_idx":
                self.m_params["imgs_idx"] = [int(argv[cnt + 1])]
                index = cnt + 1

                while index + 1 < argc and numcheck(argv[index + 1]):
                    self.m_params["imgs_idx"].append(int(argv[index + 1]))
                    index += 1

            elif option == "-train_data":
                self.m_params["train_data"] = []
                count = int(argv[cnt + 1])
                for offset in range(count):
                    self.m_params["train_data"].append(argv[cnt + offset + 2])

            elif option in self._FLOAT_OPTIONS:
                self.m_params[self._FLOAT_OPTIONS[option]] = float(argv[cnt + 1])

            elif option in self._INT_OPTIONS:
                self.m_params[self._INT_OPTIONS[option]] = int(argv[cnt + 1])

            cnt += 1

        return 0

    def getParams():
        """Retain the historical placeholder method signature."""

        return 0

    def __getitem__(self, idx):
        return self.m_params[idx]

    def __setitem__(self, key, value):
        self.m_params[key] = value

    # This class attribute is intentionally shared: the legacy scripts use it
    # as a simple global parameter store.
    m_params = {}


def main(argc, argv):
    """Print the historical parser demonstration."""

    print("argc = ", argc)

    for i in range(argc):
        print(argv[i])

    print("borderlin ========================================")
    strlist = {"m_fg": "this is fg path"}

    print(strlist)
    print(strlist["m_fg"])
    strlist["m_gt"] = "this is gt path"

    strlist["frameidx"] = [1, 2, 3, 4, 5]

    print(strlist["m_gt"])
    print(strlist["frameidx"])

    params = QParams()
    params["t"] = "test"
    params["value"] = 3
    params[6] = "test str"

    print(params["t"])
    print(params["value"])
    print(params[6])

    inputparams = QParams()
    inputparams.setParams(argc, argv)

    print(inputparams["pa_im"])
    print(numcheck(4))
    print(inputparams["imgs_idx"])

    print("borderline ==================================")
    print(inputparams["pa_im"])
    print(inputparams["pa_gt"])
    print(inputparams["pa_fg"])
    print(inputparams["pa_net"])
    print(inputparams["pa_out"])
    print(inputparams["gpuid"])
    print(inputparams["version"])
    print(inputparams["imgs_idx"])


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
