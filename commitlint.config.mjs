const legacyCommitHeaders = new Set([
  "refactor(portal): Task 1.1 - Add collapsible configuration panel to Module Inspector",
  "refactor(portal): Task 1.4 - Add parameter reference table to Module Inspector",
  "refactor(portal): Task 2.6 - Add JSON export feature to Model Inspector",
  "refactor(portal): Model Inspector adopts shared Stored Configurations panel, relocates comparison",
]);

export default {
  extends: ["@commitlint/config-conventional"],
  // Keep subject-case enforcement for new commits while allowing these exact
  // legacy headers from the shared Portal integration branch.
  ignores: [(message) => legacyCommitHeaders.has(message.split(/\r?\n/, 1)[0])],
  rules: {
    "header-max-length": [0, "always", Infinity],
    "body-max-line-length": [0, "always", Infinity],
    "footer-max-line-length": [0, "always", Infinity],
  },
};
