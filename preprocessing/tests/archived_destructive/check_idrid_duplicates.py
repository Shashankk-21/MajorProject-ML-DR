from pathlib import Path

from preprocessing.dataset import build_manifest


DATA_ROOT = Path(r"C:\MajorProject - Datasets")


print("=" * 70)
print("IDRiD IMAGE-ID DUPLICATE CHECK")
print("=" * 70)

df = build_manifest(
    "IDRiD",
    data_root=DATA_ROOT,
)

print(f"Total manifest rows: {len(df)}")
print(f"Unique image_id values: {df['image_id'].nunique()}")
print(f"Duplicate rows by image_id: {df['image_id'].duplicated(keep=False).sum()}")


# ------------------------------------------------------------
# 1. Show every duplicated image_id
# ------------------------------------------------------------

dupes = (
    df[df["image_id"].duplicated(keep=False)]
    .sort_values(["image_id", "source_split_hint"])
)

print("\n" + "=" * 70)
print("ALL DUPLICATED IMAGE IDs")
print("=" * 70)

if dupes.empty:
    print("NONE")
else:
    print(
        dupes[
            [
                "image_id",
                "source_split_hint",
                "label",
                "patient_id",
                "raw_image_locator",
            ]
        ].to_string(index=False)
    )


# ------------------------------------------------------------
# 2. Check whether duplicates occur across train/test
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DUPLICATE IDs BY SOURCE SPLIT")
print("=" * 70)

if dupes.empty:
    print("NONE")
else:
    for image_id, group in dupes.groupby("image_id"):
        splits = sorted(group["source_split_hint"].astype(str).unique())

        print(
            f"{image_id}: "
            f"splits={splits}, "
            f"rows={len(group)}"
        )


# ------------------------------------------------------------
# 3. Check whether duplicate IDs point to the same file
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DO DUPLICATES POINT TO THE SAME RAW FILE?")
print("=" * 70)

same_locator = 0
different_locator = 0

if dupes.empty:
    print("No duplicates to check.")
else:
    for image_id, group in dupes.groupby("image_id"):
        locators = set(group["raw_image_locator"].astype(str))

        if len(locators) == 1:
            same_locator += 1
            print(
                f"{image_id}: SAME raw_image_locator"
            )
        else:
            different_locator += 1
            print(
                f"{image_id}: DIFFERENT raw_image_locator(s)"
            )
            for locator in sorted(locators):
                print(f"    {locator}")


print("\nSummary:")
print(f"Duplicate IDs pointing to same raw locator    : {same_locator}")
print(f"Duplicate IDs pointing to different locators  : {different_locator}")


# ------------------------------------------------------------
# 4. Final interpretation
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

if dupes.empty:
    print("PASS: image_id is unique across the IDRiD manifest.")
elif different_locator > 0:
    print(
        "IMPORTANT: duplicate image_id values point to different "
        "raw-image locators. image_id should be made unique within "
        "the client before final-manifest/cache processing."
    )
else:
    print(
        "Duplicate image_id values exist, but they point to the same "
        "raw-image locator. Further inspection is needed before changing code."
    )