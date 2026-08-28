{ pkgs ? import <nixpkgs> {} }:
  pkgs.mkShell {
    nativeBuildInputs = with pkgs.buildPackages; [
        
      #  Installer programmer her, for eksempel:
      #vscode    # Visual Studio Code
      #nix-shell -p python313Packages.ollama inventree-part-import python313Packages.inventree python313Packages.python-dotenv
      # https://ollama.com/library/qwen2.5vl
      python313Packages.ollama
      inventree-part-import
      python313Packages.inventree
      python313Packages.opencv-python
      python313Packages.python-dotenv
      
    ];

    # Måde at sætte environment variables på! Denne måde noterer nix-store path
    # fx: VSCODE_USERDATA_DIR = ./VScodium/userdata   ---> echo $VSCODE_USERDATA_DIR -> /nix/store/2f5d...2cDs/userdata
    #VSCODE_USERDATA_DIR = ./VScode/userdata;
    #VSCODE_EXTENSION_DIR = ./VScode/extensions;

    # shellHook afvikler linux shell kode med det samme efter opstilling

    # Sørger for at holde Visual Studio code data separate for systemet.
    #shellHook = ''
    #  export VSCODE_USERDATA_DIR=~/Roadwarrior/Studie/VScode/Programmering/VScode/userdata
    #  export VSCODE_EXTENSION_DIR=~/Roadwarrior/Studie/VScode/Programmering/VScode/extensions
    #  alias vscode="code \
    #  --user-data-dir=$VSCODE_USERDATA_DIR \
    #  --extensions-dir=$VSCODE_EXTENSION_DIR"
    #  vscode
    #'';
}
