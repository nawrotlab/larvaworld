import subprocess
import pathlib

def main():
    folder=pathlib.Path(__file__).parent
    cmd = f"panel serve {str(folder/'module_tester.py')} {str(folder/'model_inspector.py')}"
    print(cmd)
    process =subprocess.Popen(
        cmd.split(),
        # shell=True,
        # check=True,
        # capture_output=True,
        # stdout=subprocess.PIPE, 
        # stderr=subprocess.PIPE,
    ) 
    process.wait()
    
    
if __name__ == "__main__":
    main()
    
    