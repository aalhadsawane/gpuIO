import subprocess
try:
    result = subprocess.run(['dpkg', '-l', 'libboost-all-dev'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    boost_installed = 'libboost-all-dev' in result.stdout.decode('utf-8')
except FileNotFoundError:
    boost_installed = False
print(f"Boost Installed: {boost_installed}")
