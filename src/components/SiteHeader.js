import Link from "next/link";
import PixelMark from "./PixelMark";

export default function SiteHeader({ links }) {
	return (
		<header className="top">
			<Link href="/" aria-label="Home">
				<PixelMark word="ZM" label="Zachary Messinger" />
			</Link>
			<nav className="caps" aria-label="Primary">
				{links.map((l) =>
					l.href.startsWith("http") ? (
						<a key={l.href} href={l.href}>
							{l.label}
						</a>
					) : (
						<Link key={l.href} href={l.href}>
							{l.label}
						</Link>
					)
				)}
			</nav>
		</header>
	);
}
