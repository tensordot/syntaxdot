{ stdenv
, defaultCrateOverrides

# Native build inputs
, cmake
}:

defaultCrateOverrides // {
  sentencepiece-sys = attr: {
    nativeBuildInputs = [ cmake ];
  };
}
