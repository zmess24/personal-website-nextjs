import React from "react";
import Link from "next/link";

function PostNav() {
	return (
		<nav className="navbar" role="navigation" aria-label="primary" id="post-nav">
			<header className="navbar-menu">
				<section className="navbar-item">
					<Link href="/" title="open">
						Home
					</Link>
				</section>
				<section className="navbar-item navbar-brand">
					<Link href="/" className="alternate">
						Zachary Messinger
					</Link>
				</section>
			</header>
		</nav>
	);
}

export default PostNav;
