// Rehype plugin for post pages.
// - Gives every h2/h3 an id and collects them into `toc` for the contents sidebar.
// - Turns remark-prism's <div class="remark-highlight"><pre>…</pre></div> into a
//   <figure class="code"> with a language label and a copy button, or into a
//   <figure class="out"> for unlabeled blocks (notebook output).

const LABELS = {
	python: "Python",
	py: "Python",
	javascript: "JavaScript",
	js: "JavaScript",
	jsx: "JSX",
	typescript: "TypeScript",
	ts: "TypeScript",
	json: "JSON",
	bash: "Shell",
	sh: "Shell",
	shell: "Shell",
	html: "HTML",
	css: "CSS",
	sql: "SQL",
	xml: "XML",
	yaml: "YAML",
};

function classList(node) {
	const c = node.properties && node.properties.className;
	if (!c) return [];
	return Array.isArray(c) ? c : String(c).split(/\s+/);
}

function textOf(node) {
	if (node.type === "text") return node.value;
	return (node.children || []).map(textOf).join("");
}

function slugify(text) {
	return text
		.toLowerCase()
		.replace(/&/g, " and ")
		.replace(/[^a-z0-9\s-]/g, "")
		.trim()
		.replace(/\s+/g, "-")
		.replace(/-+/g, "-");
}

const el = (tagName, properties, children) => ({ type: "element", tagName, properties, children });
const txt = (value) => ({ type: "text", value });

function codeFigure(pre) {
	const lang = (classList(pre).find((c) => c.startsWith("language-")) || "language-unknown").slice(9);
	if (lang === "unknown" || lang === "text" || lang === "") {
		return el("figure", { className: ["out"] }, [el("figcaption", { className: ["caps"] }, [txt("Output")]), pre]);
	}
	const label = LABELS[lang] || lang.charAt(0).toUpperCase() + lang.slice(1);
	return el("figure", { className: ["code"] }, [
		el("figcaption", {}, [
			el("span", { className: ["caps"] }, [txt(label)]),
			el("button", { className: ["cp", "caps"], type: "button", ariaLabel: `Copy ${label} code` }, [txt("Copy")]),
		]),
		pre,
	]);
}

export default function rehypePostBlocks({ toc }) {
	return (tree) => {
		const used = new Set();
		const walk = (node) => {
			if (!node.children) return;
			node.children = node.children.map((child) => {
				if (child.type !== "element") return child;
				if (child.tagName === "div" && classList(child).includes("remark-highlight")) {
					const pre = (child.children || []).find((c) => c.type === "element" && c.tagName === "pre");
					if (pre) return codeFigure(pre);
				}
				if (child.tagName === "h2" || child.tagName === "h3") {
					const text = textOf(child).trim();
					let id = slugify(text) || "section";
					let n = 2;
					while (used.has(id)) id = `${slugify(text)}-${n++}`;
					used.add(id);
					child.properties = { ...(child.properties || {}), id };
					toc.push({ id, text, depth: Number(child.tagName[1]) });
				}
				walk(child);
				return child;
			});
		};
		walk(tree);
	};
}
