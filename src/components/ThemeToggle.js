// Light and dark mode. The page follows the system setting until someone picks a theme here;
// the pick is kept in localStorage and applied before first paint by the script in _document.js.
// Which icon shows is decided in CSS, so the server and client render the same markup.
const KEY = "theme";

function currentTheme() {
	const set = document.documentElement.dataset.theme;
	if (set === "light" || set === "dark") return set;
	return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export default function ThemeToggle() {
	const toggle = () => {
		const next = currentTheme() === "dark" ? "light" : "dark";
		document.documentElement.dataset.theme = next;
		try {
			localStorage.setItem(KEY, next);
		} catch (_) {
			// Storage can be blocked; the theme still applies for this page view.
		}
	};

	return (
		<button type="button" className="tg" onClick={toggle} aria-label="Switch between light and dark mode" title="Light or dark">
			<svg className="moon" viewBox="0 0 24 24" aria-hidden="true">
				<path d="M20.5 14.6A8.5 8.5 0 0 1 9.4 3.5a8.5 8.5 0 1 0 11.1 11.1Z" />
			</svg>
			<svg className="sun" viewBox="0 0 24 24" aria-hidden="true">
				<circle cx="12" cy="12" r="4" />
				<path d="M12 2.5v2.5M12 19v2.5M4.6 4.6l1.8 1.8M17.6 17.6l1.8 1.8M2.5 12H5M19 12h2.5M4.6 19.4l1.8-1.8M17.6 6.4l1.8-1.8" />
			</svg>
		</button>
	);
}
