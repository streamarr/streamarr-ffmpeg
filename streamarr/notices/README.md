# Bundled component notices

The license texts and notices of the components that the portable builds link statically,
one directory per component. They are copied from the notice inventory that the Streamarr
transcode worker reviewed for this upstream release (streamarr/streamarr-transcode-worker,
`buildpacks/ffmpeg/notices` at 41af16aecd17edb1c2d1947e6d4edc7835d53c24, bound to source
revision 7250de8375a4959ac1c25c139ce9789fdac199fb). That inventory starts from the Linux
binaries' build configuration. The macOS build also runs the recipes in builder/scripts.d, but
it builds gettext, brotli and libpng from builder/images/macos as well, and their notices are
not in this inventory yet.

The release workflow packs this directory into every archive as `licenses/components/`.
Update it from the worker's reviewed inventory whenever the fork adopts a new upstream release.
