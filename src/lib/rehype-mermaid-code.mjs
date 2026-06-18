import { visit } from "unist-util-visit";

function textFromNode(node) {
  if (!node.children) return "";
  return node.children
    .map((child) => {
      if (child.type === "text") return child.value;
      if (child.children) return textFromNode(child);
      return "";
    })
    .join("");
}

export function rehypeMermaidCode() {
  return (tree) => {
    visit(tree, "element", (node, index, parent) => {
      if (!parent || node.tagName !== "pre") return;
      const code = node.children?.find((child) => child.tagName === "code");
      const classes = code?.properties?.className ?? [];

      if (!classes.includes("language-mermaid")) return;

      parent.children[index] = {
        type: "element",
        tagName: "div",
        properties: { className: ["mermaid"] },
        children: [{ type: "text", value: textFromNode(code) }]
      };
    });
  };
}
