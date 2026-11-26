import Head from "next/head";
import SiteHeader from "@/components/SiteHeader";
import SiteFooter from "@/components/SiteFooter";
import SwishJam from "@/components/SwishJam";
import NowCard from "@/components/NowCard";
import { NOW, SITE_URL } from "@/data/site";

// Only the projects I'm actively building. Earlier builds stay on the home page's timeline.
export default function Projects() {
	return (
		<>
			<Head>
				<title>Projects · Zachary Messinger</title>
				<meta name="description" content="What I'm building now: CLZ Labs and Rung." />
				<meta property="og:title" content="Projects · Zachary Messinger" />
				<meta property="og:url" content={`${SITE_URL}/projects`} />
			</Head>
			<SwishJam>
				<div className="wrap">
					<SiteHeader />
					<main>
						<header className="page-h">
							<h1>Projects</h1>
							<p>What I&apos;m building now. Earlier builds are on the timeline on the home page.</p>
						</header>
						<section className="split solo" aria-label="Active projects">
							{NOW.map((n) => (
								<NowCard key={n.name} n={n} />
							))}
						</section>
					</main>
					<SiteFooter />
				</div>
			</SwishJam>
		</>
	);
}
