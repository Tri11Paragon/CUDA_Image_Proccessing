{
	pkgs ? import <nixpkgs> {
		config.allowUnfree = true;
    	config.segger-jlink.acceptLicense = true;
	},
	customPkgs ? import /home/brett/my-nixpkgs {
		config.allowUnfree = true;
    	config.segger-jlink.acceptLicense = true;
	},
	...
}:
pkgs.mkShell {
	buildInputs = with pkgs; [
		customPkgs.jetbrains.clion
		gcc
		cmake
		git
	];
	propagatedBuildInputs = with pkgs; [
		(opencv.override {enableGtk3 = true; })
		openblas
	];
}
	
