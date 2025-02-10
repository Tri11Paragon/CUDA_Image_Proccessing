{ 
	lib,
	buildPythonPackage,
	fetchgit,

	setuptools,
	setuptools-scm,
	distutils,
	pkg-config,
	gdb,
	ncurses,
	numpy,
	py,
	cmake,
	pkgs ? (import <nixpkgs> { 
	    config.allowUnfree = true;
	    config.segger-jlink.acceptLicense = true; 
	})
}:
(buildPythonPackage rec {
  pname = "dearpygui";
  version = "2.0.0";
  pyproject = true;
  src = fetchgit {
		url = "https://github.com/hoffstadt/DearPyGui.git";
		fetchSubmodules = true;
        sha256 = "sha256-jGxkedygq0HC6/i/iBpwIMkqNFZdaIKCgoi2Ou0xVBg=";
  };
  preConfigure = ''
	cd ./thirdparty/cpython/
	mkdir -p build/release
	cd build/release
	../../configure
	make $jobs
	cd ../../
	cd ../../
  '';
  pypaBuildFlags = ["../"];

  build-system = [ 
	pkgs.python312Packages.build
	setuptools 
	setuptools-scm 
	distutils 
	pkg-config
  ];
  dependencies = [ setuptools py distutils ];
  nativeBuildInputs = [ 
	pkgs.python312Packages.build
	cmake
	pkgs.gcc 
  ];
  buildInputs = with pkgs; [
	xorg.libX11 
	xorg.libX11.dev
	xorg.libXcursor
	xorg.libXcursor.dev
	xorg.libXext 
	xorg.libXext.dev
	xorg.libXinerama
	xorg.libXinerama.dev 
	xorg.libXrandr 
	xorg.libXrandr.dev
	xorg.libXrender
	xorg.libXrender.dev
	xorg.libxcb
	xorg.libxcb.dev
	xorg.libXi
	xorg.libXi.dev
	libGL
	libGL.dev
	libffi
	libffi.dev
	openssl
	openssl.dev
	gdb
	gcc
	libxcrypt	
  ];	
  propagatedBuildInputs = [ ];
  nativeCheckInputs = [ ];
})
