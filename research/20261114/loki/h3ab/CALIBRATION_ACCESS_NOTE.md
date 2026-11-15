# Calibration access clarification

The frozen protocol's phrase “No N13 target banks, labels, heldout keys or target accuracy accessed during calibration” is broader than the implementation. The shared `frozen()` integrity guard reads historical artifact bytes, including existing target files, to check their SHA-256 hashes against the predeclared manifest. This does not establish filesystem isolation from target artifacts.

The calibration preparation, timing workers and count selector do not decode those target arrays, keys or labels. They use the complete N12 source bank, source-derived clues, inherited source-trained model, shared initialization arrays and measured solve times. They never compute target accuracy, and no target accuracy enters cap selection. Fresh target inputs are reserved only after the count-selection receipt is sealed, and exclude all calibration keys.

The frozen protocol and original receipts remain unchanged. The report and summary state this distinction explicitly: **source-only refers to the information used for calibration, not a filesystem access prohibition**. No scientific setting or selected cap is changed by this clarification.
