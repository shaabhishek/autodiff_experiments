from fwd_mode_tests import f1, f2, test_with_pytorch, test_with_fwdnumber

if __name__ == '__main__':
    for f in [f1, f2]:
        test_with_fwdnumber(f)
        test_with_pytorch(f)
