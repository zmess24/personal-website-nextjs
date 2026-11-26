import Link from "next/link";
import { useRouter } from "next/router";
import PixelMark from "./PixelMark";
import ThemeToggle from "./ThemeToggle";

const LINKS = [
	{ href: "/blog", label: "Blog", match: (p) => p === "/blog" || p.startsWith("/posts/") },
	{ href: "/projects", label: "Projects", match: (p) => p === "/projects" },
];

export default function SiteHeader() {
	const { pathname } = useRouter();
	return (
		<header className="top">
			<Link href="/" aria-label="Home">
				<PixelMark word="ZM" label="Zachary Messinger" />
			</Link>
			<div className="top-r">
				<nav className="caps" aria-label="Primary">
					{LINKS.map((l) => {
						const on = l.match(pathname);
						return (
							<Link key={l.href} href={l.href} className={on ? "on" : undefined} aria-current={on ? "page" : undefined}>
								{l.label}
							</Link>
						);
					})}
				</nav>
				<ThemeToggle />
			</div>
		</header>
	);
}
