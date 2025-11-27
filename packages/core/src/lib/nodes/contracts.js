/**
 * Core interfaces and contracts for the modular node definition system.
 * These interfaces define the structure for extensible, framework-agnostic node definitions.
 */
/**
 * Supported backend frameworks for model building
 */
export var BackendFramework;
(function (BackendFramework) {
    BackendFramework["PyTorch"] = "pytorch";
    BackendFramework["TensorFlow"] = "tensorflow";
})(BackendFramework || (BackendFramework = {}));
