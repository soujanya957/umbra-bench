import LightCordHero from "@/components/LightCordHero";
import Abstract from "@/components/sections/Abstract";
import DesignTool from "@/components/sections/DesignTool";
import DemoGallery from "@/components/sections/DemoGallery";
import Developers from "@/components/sections/Developers";

export default function Home() {
  return (
    <main>
      <LightCordHero />
      <Abstract />
      <DemoGallery />
      <DesignTool />
      <Developers />
      <footer className="border-t border-umbra-glow/10 px-6 py-10 text-center text-xs text-umbra-ink/40">
        UMBRA · multi-robot shadow art
      </footer>
    </main>
  );
}
