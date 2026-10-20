# Use this sprite in a game

Keep the ZIP's folder structure. Each candidate is a separate character export; give each
candidate its own folder. The PNGs and animation metadata are the source of truth.

## Godot 4

1. Extract this entire ZIP into a folder inside your Godot project.
2. Let the editor import `sheet.png`.
3. Drag `godot/character.tscn` into your scene. Position it where the character's feet belong.
4. Select the AnimatedSprite2D node and choose `walk_down`, `walk_left`, `walk_right` or `walk_up`.

`godot/sprite_frames.tres` contains the atlas regions, loop flags and frame durations.
The scene uses nearest filtering and a bottom-center origin. The scene's position starts
at (0, 0); move it into your camera's view. Preview playback starts with `walk_down`.
The remaining four still poses remain in `frames/` and in the manifest.

## Unity 6

The Unity project needs the **2D Sprite** package (`com.unity.2d.sprite`).

1. Extract this entire ZIP into a new folder under your project's `Assets/` directory.
2. Install the helper **once per project**: copy `unity/MCPBytesSpriteImporter.cs.txt` to
   `Assets/Editor/MCPBytesSpriteImporter.cs` (remove the `.txt` suffix). Keep only one copy
   of that C# helper when importing multiple candidates. Wait for Unity to compile it.
3. Select this candidate's `unity/import.json` in the Project window.
4. Choose **Tools > MCPBytes > Import selected sprite export**.
5. Drag the generated `character.prefab` into your scene. Its origin is at the feet.

The helper configures Point filtering, no mipmaps, uncompressed texture data, all frame
rectangles and bottom-center pivots. It creates four looping AnimationClips, an Animator
Controller and a SpriteRenderer prefab. The default clip is `walk_down`; choose another
state with your own controller logic or `Animator.Play("walk_left")`.

The default scale is 32 pixels per world unit. Change `pixels_per_unit` in `unity/import.json`
before importing if your game uses another scale. Each import creates a new output folder
and preserves existing sprite IDs by name; it does not overwrite your animation assets.

## Review before use

These adapters preserve the generated poses; they do not add motion. Walking uses three
distinct poses in the sequence 0, 1, 2, 1. Review identity, anatomy, clipping and transparency.
The extra jump and lying poses are still images, not complete jump or death animations.
