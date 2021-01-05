{ defaultCrateOverrides
, lib
, symlinkJoin

# Native build inputs
, cmake
, installShellFiles
, removeReferencesTo

# Build inputs
, hdf5
, libtorch-bin
}:

defaultCrateOverrides // {
  sentencepiece-sys = attrs: {
    nativeBuildInputs = [ cmake ];

    postInstall = ''
      # Binaries and shared libraries contain references to /build,
      # but we do not need them anyway.
      rm -f $lib/lib/sentencepiece-sys.out/build/src/spm_*
      rm -f $lib/lib/sentencepiece-sys.out/build/src/*.so*
    '';

  };

  syntaxdot-cli = attr: rec {
    pname = "syntaxdot";
    name = "${pname}-${attr.version}";

    buildInputs = [ libtorch-bin ];

    nativeBuildInputs = [
      installShellFiles
      removeReferencesTo
    ];

    postInstall = ''
      # We do not care for sticker2-utils as a library crate. Removing
      # the library ensures that we don't get any stray references.
      rm -rf $lib/lib

      # Install shell completions
      for shell in bash fish zsh; do
        target/bin/syntaxdot completions $shell > completions.$shell
      done

      installShellCompletion completions.{bash,fish,zsh}

      remove-references-to -t ${libtorch-bin.dev} $out/bin/syntaxdot
    '';

    disallowedReferences = [ libtorch-bin.dev ];

    meta = with lib; {
      description = "Neural sequence labeler";
      license = licenses.blueOak100;
      maintainers = with maintainers; [ danieldk ];
      platforms = platforms.all;
    };
  };

  torch-sys = attr: {
    LIBTORCH = libtorch-bin.dev;
  };
}