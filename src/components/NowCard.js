import Image from "next/image";
import PixelMark from "./PixelMark";

// A large card for something I'm building now (CLZ Labs, Rung), used on the home and Projects pages.
export default function NowCard({ n }) {
	return (
		<a className="big" href={n.url} target="_blank" rel="noreferrer">
			<span className="mk">
				{n.mark === "clz" ? <PixelMark word="CLZ" label="CLZ Labs" /> : <Image src={n.icon} alt="" width={64} height={64} />}
			</span>
			<div>
				<h2>{n.name}</h2>
				<p>{n.description}</p>
				<p className="h caps">
					{n.since} &middot; {n.host} &rarr;
				</p>
			</div>
		</a>
	);
}
